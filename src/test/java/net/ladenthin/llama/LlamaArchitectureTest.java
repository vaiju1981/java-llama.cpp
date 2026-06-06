// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.fields;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;
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
     * import anything from elsewhere in the project — neither the root API
     * package nor the {@code json} parser package.
     *
     * <p>This pins the only stackable layer relationship in jllama. The
     * traditional {@code layeredArchitecture()} 3-layer rule (Args → Json → Api)
     * was attempted and rejected: {@code json} parsers/serializers genuinely
     * depend on root-package DTOs ({@code Pair}, {@code ChatMessage},
     * {@code ContentPart}) AND the root API genuinely depends on {@code json}
     * parsers — they are <em>peers in the public API layer</em>, not a
     * stackable hierarchy. Splitting the DTOs into a dedicated
     * {@code net.ladenthin.llama.value} package would enable real layering,
     * but breaks the published public-API FQNs ({@code net.ladenthin.llama.Pair}
     * etc.) and is out of scope for an ArchUnit rule.
     *
     * <p>So the only real architectural invariant worth enforcing here is "args
     * stays a leaf" — and that is what this rule does.
     */
    @ArchTest
    static final ArchRule argsPackageIsALeaf = noClasses()
            .that()
            .resideInAPackage("net.ladenthin.llama.args..")
            .should()
            .dependOnClassesThat()
            .resideInAnyPackage("net.ladenthin.llama", "net.ladenthin.llama.json..");

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
}
