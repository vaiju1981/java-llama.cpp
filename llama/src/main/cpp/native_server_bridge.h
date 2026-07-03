// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// Declarations for the upstream server entry point (llama.cpp tools/server/server.cpp) that
// jllama's NativeServer JNI bridge (native_server.cpp) calls to run the full llama.cpp HTTP
// server — WebUI included — inside libjllama, with no separate llama-server executable.
//
//  - llama_server: upstream's renamed main (b9859 already exposes `int llama_server(int, char**)`
//    as a non-static, externally linkable function). Runs the server and blocks until shutdown,
//    returning its process-style exit code (0 = clean).
//  - llama_server_set_embedded / llama_server_request_shutdown: added by
//    patches/0006-server-embed-native-server-jni.patch so the server can run embedded in the JVM
//    (does not install process-wide signal handlers, and honors the forwarded argv instead of
//    re-deriving it from the process command line) and can be stopped out-of-band (the SIGTERM
//    path) since its server_context is local to llama_server().

int  llama_server(int argc, char ** argv);
void llama_server_set_embedded(bool embedded);
void llama_server_request_shutdown();
