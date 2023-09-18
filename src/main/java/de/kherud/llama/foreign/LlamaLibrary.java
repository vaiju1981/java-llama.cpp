package de.kherud.llama.foreign;
import com.sun.jna.Callback;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.PointerType;
import com.sun.jna.ptr.FloatByReference;
import com.sun.jna.ptr.PointerByReference;
import de.kherud.llama.foreign.llama_grammar_element.ByReference;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
/**
 * JNA Wrapper for library <b>llama</b><br>
 * This file was autogenerated by <a href="http://jnaerator.googlecode.com/">JNAerator</a>,<br>
 * a tool written by <a href="http://ochafik.com/">Olivier Chafik</a> that <a href="http://code.google.com/p/jnaerator/wiki/CreditsAndLicense">uses a few opensource projects.</a>.<br>
 * For help, please visit <a href="http://nativelibs4java.googlecode.com/">NativeLibs4Java</a> , <a href="http://rococoa.dev.java.net/">Rococoa</a>, or <a href="http://jna.dev.java.net/">JNA</a>.
 */
public class LlamaLibrary implements Library {
	public static final String JNA_LIBRARY_NAME = "llama";
	public static final NativeLibrary JNA_NATIVE_LIB = NativeLibrary.getInstance(LlamaLibrary.JNA_LIBRARY_NAME);
	static {
		Native.register(LlamaLibrary.class, LlamaLibrary.JNA_NATIVE_LIB);
	}
	public static interface llama_log_level {
		public static final int LLAMA_LOG_LEVEL_ERROR = 2;
		public static final int LLAMA_LOG_LEVEL_WARN = 3;
		public static final int LLAMA_LOG_LEVEL_INFO = 4;
	};
	public static interface llama_vocab_type {
		public static final int LLAMA_VOCAB_TYPE_SPM = 0;
		public static final int LLAMA_VOCAB_TYPE_BPE = 1;
	};
	public static interface llama_token_type {
		public static final int LLAMA_TOKEN_TYPE_UNDEFINED = 0;
		public static final int LLAMA_TOKEN_TYPE_NORMAL = 1;
		public static final int LLAMA_TOKEN_TYPE_UNKNOWN = 2;
		public static final int LLAMA_TOKEN_TYPE_CONTROL = 3;
		public static final int LLAMA_TOKEN_TYPE_USER_DEFINED = 4;
		public static final int LLAMA_TOKEN_TYPE_UNUSED = 5;
		public static final int LLAMA_TOKEN_TYPE_BYTE = 6;
	};
	public static interface llama_ftype {
		public static final int LLAMA_FTYPE_ALL_F32 = 0;
		public static final int LLAMA_FTYPE_MOSTLY_F16 = 1;
		public static final int LLAMA_FTYPE_MOSTLY_Q4_0 = 2;
		public static final int LLAMA_FTYPE_MOSTLY_Q4_1 = 3;
		public static final int LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4;
		public static final int LLAMA_FTYPE_MOSTLY_Q8_0 = 7;
		public static final int LLAMA_FTYPE_MOSTLY_Q5_0 = 8;
		public static final int LLAMA_FTYPE_MOSTLY_Q5_1 = 9;
		public static final int LLAMA_FTYPE_MOSTLY_Q2_K = 10;
		public static final int LLAMA_FTYPE_MOSTLY_Q3_K_S = 11;
		public static final int LLAMA_FTYPE_MOSTLY_Q3_K_M = 12;
		public static final int LLAMA_FTYPE_MOSTLY_Q3_K_L = 13;
		public static final int LLAMA_FTYPE_MOSTLY_Q4_K_S = 14;
		public static final int LLAMA_FTYPE_MOSTLY_Q4_K_M = 15;
		public static final int LLAMA_FTYPE_MOSTLY_Q5_K_S = 16;
		public static final int LLAMA_FTYPE_MOSTLY_Q5_K_M = 17;
		public static final int LLAMA_FTYPE_MOSTLY_Q6_K = 18;
		public static final int LLAMA_FTYPE_GUESSED = 1024;
	};
	public static interface llama_gretype {
		public static final int LLAMA_GRETYPE_END = 0;
		public static final int LLAMA_GRETYPE_ALT = 1;
		public static final int LLAMA_GRETYPE_RULE_REF = 2;
		public static final int LLAMA_GRETYPE_CHAR = 3;
		public static final int LLAMA_GRETYPE_CHAR_NOT = 4;
		public static final int LLAMA_GRETYPE_CHAR_RNG_UPPER = 5;
		public static final int LLAMA_GRETYPE_CHAR_ALT = 6;
	};
	public static final int LLAMA_MAX_DEVICES = (int)1;
	public static final long LLAMA_DEFAULT_SEED = (long)0xFFFFFFFFL;
	public static final int LLAMA_FILE_MAGIC_GGSN = (int)0x6767736e;
	public static final int LLAMA_SESSION_MAGIC = (int)0x6767736e;
	public static final int LLAMA_SESSION_VERSION = (int)1;
	public interface llama_progress_callback extends Callback {
		void apply(float progress, Pointer ctx);
	};
	public interface llama_log_callback extends Callback {
		void apply(int level, String text, Pointer user_data);
	};
	public interface llama_beam_search_callback_fn_t extends Callback {
		void apply(Pointer callback_data, de.kherud.llama.foreign.llama_beams_state.ByValue llama_beams_state1);
	};
	public static native de.kherud.llama.foreign.llama_context_params.ByValue llama_context_default_params();
	public static native llama_model_quantize_params.ByValue llama_model_quantize_default_params();
	public static native void llama_backend_init(byte numa);
	public static native void llama_backend_free();
	public static native LlamaLibrary.llama_model llama_load_model_from_file(String path_model, de.kherud.llama.foreign.llama_context_params.ByValue params);
	public static native void llama_free_model(LlamaLibrary.llama_model model);
	public static native LlamaLibrary.llama_context llama_new_context_with_model(LlamaLibrary.llama_model model, de.kherud.llama.foreign.llama_context_params.ByValue params);
	public static native void llama_free(LlamaLibrary.llama_context ctx);
	public static native long llama_time_us();
	public static native int llama_max_devices();
	public static native byte llama_mmap_supported();
	public static native byte llama_mlock_supported();
	public static native int llama_n_vocab(LlamaLibrary.llama_context ctx);
	public static native int llama_n_ctx(LlamaLibrary.llama_context ctx);
	public static native int llama_n_ctx_train(LlamaLibrary.llama_context ctx);
	public static native int llama_n_embd(LlamaLibrary.llama_context ctx);
	public static native int llama_vocab_type(LlamaLibrary.llama_context ctx);
	public static native int llama_model_n_vocab(LlamaLibrary.llama_model model);
	public static native int llama_model_n_ctx(LlamaLibrary.llama_model model);
	public static native int llama_model_n_ctx_train(LlamaLibrary.llama_model model);
	public static native int llama_model_n_embd(LlamaLibrary.llama_model model);
	public static native int llama_model_desc(LlamaLibrary.llama_model model, byte[] buf, NativeSize buf_size);
	public static native long llama_model_size(LlamaLibrary.llama_model model);
	public static native long llama_model_n_params(LlamaLibrary.llama_model model);
	public static native int llama_model_quantize(String fname_inp, String fname_out, llama_model_quantize_params params);
	public static native int llama_apply_lora_from_file(LlamaLibrary.llama_context ctx, String path_lora, String path_base_model, int n_threads);
	public static native int llama_model_apply_lora_from_file(LlamaLibrary.llama_model model, String path_lora, String path_base_model, int n_threads);
	public static native int llama_get_kv_cache_token_count(LlamaLibrary.llama_context ctx);
	public static native void llama_set_rng_seed(LlamaLibrary.llama_context ctx, int seed);
	public static native NativeSize llama_get_state_size(LlamaLibrary.llama_context ctx);
	public static native NativeSize llama_copy_state_data(LlamaLibrary.llama_context ctx, ByteBuffer dst);
	public static native NativeSize llama_set_state_data(LlamaLibrary.llama_context ctx, ByteBuffer src);
	public static native byte llama_load_session_file(LlamaLibrary.llama_context ctx, String path_session, IntBuffer tokens_out, NativeSize n_token_capacity, NativeSizeByReference n_token_count_out);
	public static native byte llama_save_session_file(LlamaLibrary.llama_context ctx, String path_session, IntBuffer tokens, NativeSize n_token_count);
	public static native int llama_eval(LlamaLibrary.llama_context ctx, IntBuffer tokens, int n_tokens, int n_past, int n_threads);
	public static native int llama_eval_embd(LlamaLibrary.llama_context ctx, FloatBuffer embd, int n_tokens, int n_past, int n_threads);
	public static native int llama_eval_export(LlamaLibrary.llama_context ctx, String fname);
	public static native FloatByReference llama_get_logits(LlamaLibrary.llama_context ctx);
	public static native FloatByReference llama_get_embeddings(LlamaLibrary.llama_context ctx);
	public static native String llama_token_get_text(LlamaLibrary.llama_context ctx, int token);
	public static native float llama_token_get_score(LlamaLibrary.llama_context ctx, int token);
	public static native int llama_token_get_type(LlamaLibrary.llama_context ctx, int token);
	public static native int llama_token_bos(LlamaLibrary.llama_context ctx);
	public static native int llama_token_eos(LlamaLibrary.llama_context ctx);
	public static native int llama_token_nl(LlamaLibrary.llama_context ctx);
	public static native int llama_tokenize(LlamaLibrary.llama_context ctx, String text, int text_len, IntBuffer tokens, int n_max_tokens, byte add_bos);
	public static native int llama_tokenize_with_model(LlamaLibrary.llama_model model, String text, int text_len, IntBuffer tokens, int n_max_tokens, byte add_bos);
	public static native int llama_token_to_piece(LlamaLibrary.llama_context ctx, int token, ByteBuffer buf, int length);
	public static native int llama_token_to_piece_with_model(LlamaLibrary.llama_model model, int token, String buf, int length);
	public static native LlamaLibrary.llama_grammar llama_grammar_init(Pointer rules, NativeSize n_rules, NativeSize start_rule_index);
	public static native void llama_grammar_free(LlamaLibrary.llama_grammar grammar);
	public static native LlamaLibrary.llama_grammar llama_grammar_copy(LlamaLibrary.llama_grammar grammar);
	public static native void llama_sample_repetition_penalty(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, IntBuffer last_tokens, NativeSize last_tokens_size, float penalty);
	public static native void llama_sample_frequency_and_presence_penalties(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, IntBuffer last_tokens, NativeSize last_tokens_size, float alpha_frequency, float alpha_presence);
	public static native void llama_sample_classifier_free_guidance(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, LlamaLibrary.llama_context guidance_ctx, float scale);
	public static native void llama_sample_softmax(LlamaLibrary.llama_context ctx, llama_token_data_array candidates);
	public static native void llama_sample_top_k(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, int k, NativeSize min_keep);
	public static native void llama_sample_top_p(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float p, NativeSize min_keep);
	public static native void llama_sample_tail_free(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float z, NativeSize min_keep);
	public static native void llama_sample_typical(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float p, NativeSize min_keep);
	public static native void llama_sample_temperature(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float temp);
	public static native void llama_sample_grammar(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, LlamaLibrary.llama_grammar grammar);
	public static native int llama_sample_token_mirostat(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float tau, float eta, int m, FloatBuffer mu);
	public static native int llama_sample_token_mirostat_v2(LlamaLibrary.llama_context ctx, llama_token_data_array candidates, float tau, float eta, FloatBuffer mu);
	public static native int llama_sample_token_greedy(LlamaLibrary.llama_context ctx, llama_token_data_array candidates);
	public static native int llama_sample_token(LlamaLibrary.llama_context ctx, llama_token_data_array candidates);
	public static native void llama_grammar_accept_token(LlamaLibrary.llama_context ctx, LlamaLibrary.llama_grammar grammar, int token);
	public static native void llama_beam_search(LlamaLibrary.llama_context ctx, LlamaLibrary.llama_beam_search_callback_fn_t callback, Pointer callback_data, NativeSize n_beams, int n_past, int n_predict, int n_threads);
	public static native de.kherud.llama.foreign.llama_timings.ByValue llama_get_timings(LlamaLibrary.llama_context ctx);
	public static native void llama_print_timings(LlamaLibrary.llama_context ctx);
	public static native void llama_reset_timings(LlamaLibrary.llama_context ctx);
	public static native String llama_print_system_info();
	public static native void llama_log_set(LlamaLibrary.llama_log_callback log_callback, Pointer user_data);
	public static native void llama_dump_timing_info_yaml(PointerByReference stream, LlamaLibrary.llama_context ctx);
	/** Pointer to unknown (opaque) type */
	public static class llama_grammar extends PointerType {
		public llama_grammar(Pointer address) {
			super(address);
		}
		public llama_grammar() {
			super();
		}
	};
	/** Pointer to unknown (opaque) type */
	public static class FILE extends PointerType {
		public FILE(Pointer address) {
			super(address);
		}
		public FILE() {
			super();
		}
	};
	/** Pointer to unknown (opaque) type */
	public static class llama_model extends PointerType {
		public llama_model(Pointer address) {
			super(address);
		}
		public llama_model() {
			super();
		}
	};
	/** Pointer to unknown (opaque) type */
	public static class llama_context extends PointerType {
		public llama_context(Pointer address) {
			super(address);
		}
		public llama_context() {
			super();
		}
	};
}
