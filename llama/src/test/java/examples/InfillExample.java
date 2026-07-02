// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package examples;

import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LlamaOutput;

public class InfillExample {

    public static void main(String... args) {
        ModelParameters modelParams =
                new ModelParameters().setModel("models/codellama-7b.Q2_K.gguf").setGpuLayers(43);

        String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
        String suffix = "\n    return result\n";
        try (LlamaModel model = new LlamaModel(modelParams)) {
            System.out.print(prefix);
            InferenceParameters inferParams =
                    new InferenceParameters("").withInputPrefix(prefix).withInputSuffix(suffix);
            for (LlamaOutput output : model.generate(inferParams)) {
                System.out.print(output);
            }
            System.out.print(suffix);
        }
    }
}
