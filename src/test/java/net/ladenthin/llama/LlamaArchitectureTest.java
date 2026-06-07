// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.fields;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;
import static com.tngtech.archunit.library.Architectures.layeredArchitecture;
import static com.tngtech.archunit.library.dependencies.SlicesRuleDefinition.slices;

import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.lang.ArchRule;
import java.util.Random;
import org.slf4j.Logger;

@AnalyzeClasses(packages = "net.ladenthin.llama", importOptions = ImportOption.DoNotIncludeTests.class)
public class LlamaArchitectureTest {

    /**
     * Production code must not use java.util.logging directly; all logging goes through SLF4J.
     */
    @ArchTest
    static final ArchRule noJavaUtilLogging = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat()
            .resideInAPackage("java.util.logging..");

    /**
     * Test-framework classes must not appear in production code.
     */
    @ArchTest
    static final ArchRule noTestFrameworksInProduction = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage("org.junit..", "net.jqwik..", "com.tngtech.archunit..");

    /**
     * Every SLF4J {@link Logger} field follows the {@code private static final} idiom.
     */
    @ArchTest
    static final ArchRule loggersArePrivateStaticFinal = fields().that()
            .haveRawType(Logger.class)
            .should()
            .bePrivate()
            .andShould()
            .beStatic()
            .andShould()
            .beFinal();

    /**
     * No package cycles between sub-packages. Catches design drift where a leaf
     * package starts importing from its parent or sibling.
     */
    @ArchTest
    static final ArchRule noPackageCycles =
            slices().matching("net.ladenthin.llama.(*)..").should().beFreeOfCycles();

    /**
     * The {@code args} sub-package is a true leaf: pure enums / constants
     * ({@code Sampler}, {@code PoolingType}, {@code ModelFlag}, …). It must not
     * import anything from elsewhere in the project. Subsumed by the
     * {@link #layeredArchitecture} rule below (args is in the Foundation layer),
     * but kept as a precise, fast-failing guard for this specific leaf.
     */
    @ArchTest
    static final ArchRule argsPackageIsALeaf = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama.args..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage(
                    "net.ladenthin.llama",
                    "net.ladenthin.llama.callback..",
                    "net.ladenthin.llama.exception..",
                    "net.ladenthin.llama.json..",
                    "net.ladenthin.llama.loader..",
                    "net.ladenthin.llama.parameters..",
                    "net.ladenthin.llama.value..");

    /**
     * Strict layered architecture. The flat root package was split into layered packages so the
     * boundaries align with packages; dependencies flow strictly top-to-bottom:
     *
     * <pre>
     *   Api          net.ladenthin.llama          (LlamaModel, Session, iterators)
     *   Loader       loader                       (native-lib loading, OS/process infra)
     *   Marshalling  json, parameters             (response parsers, parameter builders/serializer)
     *   Foundation   value, callback, exception, args
     * </pre>
     *
     * <p>The DTO/parser tangle that previously blocked this rule (see the git history of the
     * {@code argsPackageIsALeaf} comment) was resolved by extracting the value types into
     * {@code value}, moving {@code TimingsLogger} into {@code json} (its only consumer) and
     * {@code ParameterJsonSerializer} into {@code parameters} (a parameter serializer), and
     * moving {@code ChatRequest} into {@code parameters} (it carries an {@code InferenceParameters}
     * customizer). {@code json} and {@code parameters} are peers in the Marshalling layer.
     * {@code consideringOnlyDependenciesInLayers()} ignores external libraries.
     */
    @ArchTest
    static final ArchRule layeredArchitecture = layeredArchitecture()
            .consideringOnlyDependenciesInLayers()
            .layer("Api")
            .definedBy("net.ladenthin.llama")
            .layer("Loader")
            .definedBy("net.ladenthin.llama.loader..")
            .layer("Marshalling")
            .definedBy("net.ladenthin.llama.json..", "net.ladenthin.llama.parameters..")
            .layer("Foundation")
            .definedBy(
                    "net.ladenthin.llama.value..",
                    "net.ladenthin.llama.callback..",
                    "net.ladenthin.llama.exception..",
                    "net.ladenthin.llama.args..")
            .whereLayer("Api")
            .mayNotBeAccessedByAnyLayer()
            .whereLayer("Loader")
            .mayOnlyBeAccessedByLayers("Api")
            .whereLayer("Marshalling")
            .mayOnlyBeAccessedByLayers("Api", "Loader")
            .whereLayer("Foundation")
            .mayOnlyBeAccessedByLayers("Api", "Loader", "Marshalling");

    /**
     * Production code must not import unsupported / internal JDK packages.
     * These are not part of the Java SE API and may change or disappear without notice.
     * {@code OSInfo} is vendored from xerial/sqlite-jdbc and was already audited;
     * if it ever pulls in sun.*, this rule fails and forces a re-audit.
     */
    @ArchTest
    static final ArchRule noInternalJdkImports = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage("sun..", "com.sun..", "jdk.internal..");

    /**
     * Public mutable state forbidden: any non-static field declared
     * {@code public} must also be {@code final}. {@link LlamaOutput} is an
     * immutable value class with {@code public final} fields — that pattern
     * remains allowed because the fields ARE final.
     */
    @ArchTest
    static final ArchRule noPublicMutableFields =
            fields().that().arePublic().and().areNotStatic().should().beFinal();

    /**
     * Production code must not call {@link System#exit(int)}; throw an exception instead.
     */
    @ArchTest
    static final ArchRule noSystemExit = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callMethod(System.class, "exit", int.class)
            .allowEmptyShould(true);

    /**
     * Production code must not construct {@link java.util.Random}; {@code Random} is a non-cryptographic
     * PRNG (CWE-338). Use {@link java.security.SecureRandom} or {@link java.util.concurrent.ThreadLocalRandom}
     * depending on whether cryptographic strength or thread-local fast jitter is needed.
     */
    @ArchTest
    static final ArchRule noNewRandom = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callConstructor(Random.class)
            .orShould()
            .callConstructor(Random.class, long.class)
            .allowEmptyShould(true);

    /**
     * Production code must not call {@link Thread#sleep(long)} / {@link Thread#sleep(long, int)};
     * prefer {@link java.util.concurrent.BlockingQueue#poll(long, java.util.concurrent.TimeUnit)} or
     * {@link java.util.concurrent.locks.Condition#await(long, java.util.concurrent.TimeUnit)}.
     */
    @ArchTest
    static final ArchRule noThreadSleep = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama..")
            .should()
            .callMethod(Thread.class, "sleep", long.class)
            .orShould()
            .callMethod(Thread.class, "sleep", long.class, int.class)
            .allowEmptyShould(true);

    /**
     * Per-module banned import: the foundation contracts ({@code args}, {@code callback},
     * {@code exception}) and the {@code loader} infrastructure must stay free of the Jackson
     * JSON library ({@code com.fasterxml.jackson..}). JSON marshalling is the job of
     * {@code value} / {@code json} / {@code parameters} (and the root {@code Api}, which drives
     * them); these layers carry only plain typed data and native-loading logic.
     */
    @ArchTest
    static final ArchRule jacksonBannedFromContractsAndLoader = noClasses()
            .that()
            .resideInAnyPackage(
                    "net.ladenthin.llama.args..",
                    "net.ladenthin.llama.callback..",
                    "net.ladenthin.llama.exception..",
                    "net.ladenthin.llama.loader..")
            .should()
            .dependOnClassesThat()
            .resideInAPackage("com.fasterxml.jackson..")
            .allowEmptyShould(true);
}
