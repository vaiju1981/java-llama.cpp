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

// Upstream server headers must precede jni_helpers.hpp (include order rule, see CLAUDE.md);
// they provide server_context so the attach path can reach a LlamaModel's jllama_context.
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
#include "jni_helpers.hpp"

#include "native_server_bridge.h"

#include <jni.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
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

// Standard-UTF8 extraction used for argv/commands (declared here; defined below).
static std::string jstring_to_utf8(JNIEnv *env, jstring js);

// Copies the forwarded Java argv into srv->args/argv with a synthetic argv[0]. The argv pointers
// reference the std::string storage in `args`, which is filled once (with reserve) and never
// mutated afterwards, so the pointers stay valid for the worker's lifetime.
void fill_native_server_args(JNIEnv *env, jobjectArray jargs, native_server *srv) {
    const jsize n = (jargs != nullptr) ? env->GetArrayLength(jargs) : 0;
    srv->args.reserve(static_cast<size_t>(n) + 1);
    srv->args.emplace_back("llama-server"); // argv[0]
    for (jsize i = 0; i < n; ++i) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(jargs, i));
        if (js != nullptr) {
            srv->args.emplace_back(jstring_to_utf8(env, js));
            env->DeleteLocalRef(js);
        } else {
            srv->args.emplace_back("");
        }
    }

    srv->argv.reserve(srv->args.size());
    for (auto &arg : srv->args) {
        srv->argv.push_back(const_cast<char *>(arg.c_str()));
    }
}

// Throws net.ladenthin.llama.exception.LlamaException with the given message (best-effort: if the
// class cannot be resolved the pending NoClassDefFoundError is surfaced instead).
void throw_llama_exception(JNIEnv *env, const char *message) {
    jclass exception_class = env->FindClass("net/ladenthin/llama/exception/LlamaException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message);
    }
}

// Standard-UTF8 extraction (mirrors jllama.cpp parse_jstring): GetStringUTFChars yields
// Modified-UTF8 (CESU-8), which mangles non-BMP chars (emoji, CJK extensions) in argv such as
// model paths. We go through String.getBytes("UTF-8") instead (H1).
std::string jstring_to_utf8(JNIEnv *env, jstring js) {
    if (js == nullptr) {
        return "";
    }
    // Resolve and cache the JNI class/method/field IDs once (server-startup is the only caller,
    // but it can run many times — e.g. once per forwarded argv element). FindClass returns a
    // frame-local ref, so promote the reused handles to global refs for caching (MED #6).
    static jclass str_cls = nullptr;
    static jmethodID get_bytes = nullptr;
    static jobject utf8 = nullptr;
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        jclass local = env->FindClass("java/lang/String");
        if (local == nullptr) {
            return;
        }
        jmethodID gb = env->GetMethodID(local, "getBytes", "(Ljava/nio/charset/Charset;)[B");
        jclass charset_cls = env->FindClass("java/nio/charset/StandardCharsets");
        jfieldID utf8_f =
            charset_cls ? env->GetStaticFieldID(charset_cls, "UTF_8", "Ljava/nio/charset/Charset;") : nullptr;
        if (gb == nullptr || charset_cls == nullptr || utf8_f == nullptr) {
            if (charset_cls != nullptr) {
                env->DeleteLocalRef(charset_cls);
            }
            env->DeleteLocalRef(local);
            return;
        }
        jobject u = env->GetStaticObjectField(charset_cls, utf8_f);
        env->DeleteLocalRef(charset_cls);
        str_cls = static_cast<jclass>(env->NewGlobalRef(local));
        get_bytes = gb;
        utf8 = env->NewGlobalRef(u);
        env->DeleteLocalRef(local);
        env->DeleteLocalRef(u);
    });
    if (str_cls == nullptr || get_bytes == nullptr || utf8 == nullptr) {
        return "";
    }

    jbyteArray bytes = (jbyteArray)env->CallObjectMethod(js, get_bytes, utf8);
    std::string out;
    if (bytes != nullptr) {
        const jsize blen = env->GetArrayLength(bytes);
        jbyte *elems = env->GetByteArrayElements(bytes, nullptr);
        if (elems != nullptr) {
            out.assign(reinterpret_cast<char *>(elems), static_cast<size_t>(blen));
            env->ReleaseByteArrayElements(bytes, elems, JNI_ABORT);
        }
        env->DeleteLocalRef(bytes);
    }
    return out;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startNativeServer(JNIEnv *env, jclass,
                                                                                       jobjectArray jargs) {
    auto *srv = new native_server();
    fill_native_server_args(env, jargs, srv);

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

JNIEXPORT jlong JNICALL Java_net_ladenthin_llama_server_NativeServer_startAttachedNativeServer(JNIEnv *env, jclass,
                                                                                               jobject jmodel,
                                                                                               jobjectArray jargs) {
    if (jmodel == nullptr) {
        throw_llama_exception(env, "model must not be null");
        return 0;
    }
    // Read the LlamaModel's native context handle directly (the same "ctx" long field jllama.cpp
    // caches in JNI_OnLoad; this TU resolves it itself to stay decoupled from those globals).
    jclass model_class = env->GetObjectClass(jmodel);
    jfieldID ctx_field = env->GetFieldID(model_class, "ctx", "J");
    if (ctx_field == nullptr) {
        return 0; // NoSuchFieldError already pending
    }
    const jlong model_handle = env->GetLongField(jmodel, ctx_field);
    if (model_handle == 0) {
        throw_llama_exception(env, "model is not loaded (or already closed)");
        return 0;
    }
    auto *jctx = reinterpret_cast<jllama_context *>(model_handle); // NOLINT(*-no-int-to-ptr)

    // Contract (M6): llama_server_attach drives the HTTP frontend on a NEW worker thread but
    // must NOT start a second task-processing loop on the shared server_context — the model's
    // own worker (spawned in load_model_impl) already runs start_loop(). Likewise,
    // stopNativeServer's llama_server_request_shutdown() must terminate only the HTTP frontend,
    // not the attached model's worker, so the LlamaModel remains usable after the server closes
    // (documented order: close server first, then model). If a future upstream change breaks
    // either assumption, this attach path needs revisiting.
    auto *srv = new native_server();
    fill_native_server_args(env, jargs, srv);

    // The attach entry always parses the forwarded argv; set the embedded flag anyway so any
    // shared embedded-mode behavior in server.cpp stays consistent with startNativeServer.
    llama_server_set_embedded(true);

    server_context *ctx_server = &jctx->server;
    // Take a jllama_context user reference for the lifetime of the attach worker so that
    // LlamaModel.close() cannot free jctx (and thus ctx_server) out from under the running
    // HTTP frontend (use-after-free, MED #2). delete() in jllama.cpp waits on the user-count
    // condition variable, so close() blocks until this thread exits.
    jctx->users.fetch_add(1, std::memory_order_relaxed);
    srv->worker = std::thread([srv, ctx_server, jctx]() {
        srv->exit_code = llama_server_attach(static_cast<int>(srv->argv.size()), srv->argv.data(), *ctx_server);
        srv->finished.store(true);
        release_jllama_context_impl(jctx);
    });

    return reinterpret_cast<jlong>(srv);
}

JNIEXPORT void JNICALL Java_net_ladenthin_llama_server_NativeServer_setWorkerCommandNative(JNIEnv *env, jclass,
                                                                                           jstring jcommand) {
    // Sets/clears LLAMA_SERVER_WORKER_CMD in the process environment, which the router-mode
    // model manager (server-models.cpp, patches/0008) reads when spawning worker instances.
    std::string value;
    if (jcommand != nullptr) {
        value = jstring_to_utf8(env, jcommand);
    }
#if defined(_WIN32)
    _putenv_s("LLAMA_SERVER_WORKER_CMD", value.c_str()); // empty value removes the variable
#else
    if (value.empty()) {
        unsetenv("LLAMA_SERVER_WORKER_CMD");
    } else {
        setenv("LLAMA_SERVER_WORKER_CMD", value.c_str(), 1);
    }
#endif
}

} // extern "C"
