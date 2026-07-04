// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// JNI bridge for net.ladenthin.llama.server.NativeServer: runs the full upstream llama.cpp HTTP
// server (llama_server(), including its embedded WebUI) inside libjllama, driven over JNI. The
// argv is forwarded verbatim from Java, so every llama-server flag is supported. This is an
// independent server lifecycle (it loads its own model from the argv), distinct from LlamaModel
// and the Java-side OpenAiCompatServer.
//
// Only ONE native server may run per process: server.cpp keeps its shutdown_handler /
// is_terminating state in file-scope globals, so a second concurrent llama_server() would clobber
// them. NativeServer enforces this on the Java side.

#include "native_server_bridge.h"

#include <jni.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {

// Owns the argv storage for the lifetime of the running server plus the worker thread that runs
// llama_server(). The argv pointers reference the std::string storage in `args`, which is filled
// once (with reserve) and never mutated afterwards, so the pointers stay valid.
struct native_server {
    std::vector<std::string> args; // args[0] is the program name ("llama-server")
    std::vector<char *> argv;      // points into `args`
    std::thread worker;
    std::atomic<bool> finished{false};
    int exit_code = -1;
};

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startNativeServer(JNIEnv *env, jclass,
                                                                                       jobjectArray jargs) {
    auto *srv = new native_server();

    const jsize n = (jargs != nullptr) ? env->GetArrayLength(jargs) : 0;
    srv->args.reserve(static_cast<size_t>(n) + 1);
    srv->args.emplace_back("llama-server"); // argv[0]
    for (jsize i = 0; i < n; ++i) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
        if (js != nullptr) {
            const char *chars = env->GetStringUTFChars(js, nullptr);
            srv->args.emplace_back(chars != nullptr ? chars : "");
            if (chars != nullptr) {
                env->ReleaseStringUTFChars(js, chars);
            }
            env->DeleteLocalRef(js);
        } else {
            srv->args.emplace_back("");
        }
    }

    srv->argv.reserve(srv->args.size());
    for (auto &arg : srv->args) {
        srv->argv.push_back(const_cast<char *>(arg.c_str()));
    }

    // Embedded mode: no process signal handlers, honor the forwarded argv (see patches/0006).
    llama_server_set_embedded(true);

    srv->worker = std::thread([srv]() {
        srv->exit_code = llama_server(static_cast<int>(srv->argv.size()), srv->argv.data());
        srv->finished.store(true);
    });

    return reinterpret_cast<jlong>(srv);
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_stopNativeServer(JNIEnv *, jclass, jlong handle) {
    auto *srv = reinterpret_cast<native_server *>(handle);
    if (srv == nullptr) {
        return;
    }
    // Signal shutdown, retrying until the worker actually returns: a stop issued before the server
    // finished starting (shutdown_handler not yet installed by llama_server) would otherwise be
    // lost. Once the handler is installed the first signal takes effect; if the model failed to
    // load, llama_server has already returned and `finished` is set.
    while (!srv->finished.load()) {
        llama_server_request_shutdown();
        if (srv->finished.load()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (srv->worker.joinable()) {
        srv->worker.join();
    }
    delete srv;
}

JNIEXPORT jboolean JNICALL Java_net_ladenthin_llama_server_NativeServer_isRunningNative(JNIEnv *, jclass,
                                                                                        jlong handle) {
    auto *srv = reinterpret_cast<native_server *>(handle);
    return (srv != nullptr && !srv->finished.load()) ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"
