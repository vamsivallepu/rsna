#!/usr/bin/env python3
"""Materialize the self-contained P8 blend canary Kaggle script."""

from pathlib import Path


COMMON = Path("scripts/p2_submission.py")
SOURCE = Path("scripts/p4_blend_submission.py")
OUTPUT = Path("kaggle/rsna-p8-blend-canary/rsna-p8-blend-canary.py")
IMPORT_START = "# BEGIN COMMON IMPORTS"
IMPORT_END = "# END COMMON IMPORTS"
P6_SWITCH = "ENABLE_P6 = False  # The P6 deployment builder changes this immutable switch."
P8_TEACHER_MANIFEST_SHA256 = (
    "c9fa60800e450c4bb41e6d31cc3a7d8cdb74b1d1367fb32c990200a64e5bc85f"
)
P4_MANIFEST_SHA256 = {
    0: "6ed454243a6e3ffead8566edb86bb4c7ecc3be668a11fa0f53bd0c093d001932",
    1: "9541ca0158a96c2350f4fbd94d10ad385cff06d6d8457c2f92b57fdd9a1bb4d5",
    2: "93135d2849dc891115fd256472729cc817f86ea2ba79c7f8d5a1553da66aebac",
    3: "9894605caa4ae2c0d93f5d11c8766a2256f20e14298821f40dc7b1df9e83867c",
    4: "e592fa0e53cabe35819489e733bdff0885068c5756f65e541d5fee7da6b1c228",
}
P8_MANIFEST_SHA256 = {
    0: "f242c61269bff702fe601e35f6bc1bd7267f7d5f94c265ce684e665a4b37baf6",
    1: "1120c6173b9044ca84c035e47070cb5e79e40484eddea158cdd114f809eb10eb",
    2: "78925624ab4cebbe7044da362e1015c4d39b0ab514f04625f3d031f2a952c04d",
    3: "4d7f4a4090b7eb00517f29b8ea88822a0b8871519a593b64fb464fdaa71cd095",
    4: "94bacb27bc1ffa637ba3adbad6be0c960f19f69a0b59563a2b1fd6efe9f7ac2b",
}
P4_CHECKPOINT_SHA256 = {
    0: "340576ef981922173e056ced97c98f34529bcaaeec4831ea41c6c308638782ee",
    1: "bd11c89940f93cf819f3244387ffe16b49216180d5bc529431b13daa078bd0d9",
    2: "49340b64c4fd1dee9253e5712ae776b14d3805ddec73b4195b0b55caca38fae5",
    3: "16bac1a05b81fba82e2fa1e2f49b4a660de0e616662894f0a63707bdd1b89bd7",
    4: "596b267f51f52bfe489034dadaa7759342e271825834def1f4aab7a05ed069b5",
}
P8_CHECKPOINT_SHA256 = {
    0: "5883762c8f4161e68eea33766aa737b34b772b180e18e75572937eeabf366c5f",
    1: "68df88b583cca94d6f777476f0b2696c9fcd850591d863cc0255a85f248003d0",
    2: "b2b42a0c44f83986839f90ec4547038ac0ca9705a30923be3da1e209c5196498",
    3: "a2c1600b173a4abc3b94e339e92de0db92e93a0d3a5a456db1f415e0c0db3639",
    4: "d0a154d4ab0844793c195ac92beba5fb036c9ea51dfba20bdef99194fdcb868f",
}


def replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new, 1)


def main() -> int:
    common = COMMON.read_text()
    marker = "\ndef main() -> int:"
    assert marker in common
    common_prefix = common.split(marker, maxsplit=1)[0].rstrip()

    source = SOURCE.read_text()
    assert source.count(IMPORT_START) == source.count(IMPORT_END) == 1
    source = replace_once(source, P6_SWITCH, "ENABLE_P6 = True")
    before, remainder = source.split(IMPORT_START, maxsplit=1)
    _, after = remainder.split(IMPORT_END, maxsplit=1)
    deployment = before.rstrip() + "\n" + after.lstrip("\n")

    # The P8 canary deliberately changes only the five end-to-end fold sources
    # and frozen OOF constants. The proven P2/P6 path, geometry, batching, and
    # dual-device scheduler remain byte-derived from the promoted P6 source.
    deployment = deployment.replace("P4", "P8").replace("p4", "p8")
    deployment = replace_once(
        deployment,
        "rsna-knee-p8-coatnet-e2e-fold-",
        "rsna-knee-p8-noisy-student-fold-",
    )
    deployment = replace_once(
        deployment,
        "rsna-p8-coatnet-e2e-v1",
        "rsna-p8-noisy-student-v1",
    )
    deployment = replace_once(
        deployment,
        "P8-COATNET-E2E-FOLD-",
        "P8-NOISY-STUDENT-FOLD-",
    )
    for fold in range(5):
        deployment = replace_once(
            deployment,
            P4_MANIFEST_SHA256[fold],
            P8_MANIFEST_SHA256[fold],
        )
        deployment = replace_once(
            deployment,
            P4_CHECKPOINT_SHA256[fold],
            P8_CHECKPOINT_SHA256[fold],
        )
    deployment = replace_once(
        deployment,
        "P8_OOF_MACRO_AUC = 0.89116089660126",
        "P8_OOF_MACRO_AUC = 0.9029118074637345",
    )
    deployment = replace_once(
        deployment,
        "BLEND_OOF_MACRO_AUC = 0.8958210633119559",
        "BLEND_OOF_MACRO_AUC = 0.9022377777850243",
    )
    deployment = replace_once(
        deployment,
        "P6_BLEND_OOF_MACRO_AUC = 0.9054205862136073",
        "P6_BLEND_OOF_MACRO_AUC = 0.9097982866758175",
    )
    deployment = replace_once(
        deployment,
        "ENABLE_P6 = True\n",
        (
            "ENABLE_P6 = True\n"
            f'EXPECTED_P8_TEACHER_MANIFEST_SHA256 = "{P8_TEACHER_MANIFEST_SHA256}"\n'
        ),
    )
    deployment = replace_once(
        deployment,
        'assert manifest["fold"] == fold and manifest["best_epoch"] == 8\n'
        '        assert manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE',
        'assert manifest["fold"] == fold and manifest["best_epoch"] == 8\n'
        '        assert manifest["cache_signature_sha256"] == EXPECTED_CACHE_SIGNATURE\n'
        '        assert manifest["teacher_manifest_sha256"] == (\n'
        '            EXPECTED_P8_TEACHER_MANIFEST_SHA256\n'
        '        )\n'
        '        assert manifest["checkpoint_selection_gold_rows"] == 0',
    )
    deployment = replace_once(
        deployment,
        '"checkpoint_sha256": EXPECTED_P8_CHECKPOINT_SHA256[fold],',
        '"checkpoint_sha256": EXPECTED_P8_CHECKPOINT_SHA256[fold],\n'
        '                "teacher_manifest_sha256": p8_manifests[fold][\n'
        '                    "teacher_manifest_sha256"\n'
        '                ],\n'
        '                "teacher_target_sha256": p8_manifests[fold][\n'
        '                    "teacher_target_sha256"\n'
        '                ],',
    )
    deployment = replace_once(
        deployment,
        '"p8": "mean probabilities from five end-to-end P8 folds",',
        '"p8": "mean probabilities from five fold-safe noisy-student folds",',
    )
    deployment = replace_once(
        deployment,
        '"weights": (\n'
        '                {"p2_p8_equal_rank": 0.75, "p6_rank": 0.25}',
        '"formula": "0.375 rank(P2) + 0.375 rank(P8) + 0.25 rank(P6)",\n'
        '            "weights": (\n'
        '                {"p2_p8_equal_rank": 0.75, "p6_rank": 0.25}',
    )
    deployment = replace_once(
        deployment,
        'output_prefix = "p6_blend" if ENABLE_P6 else "p8_blend"',
        'output_prefix = "p8_blend"',
    )
    deployment = replace_once(
        deployment,
        'effective_run_id = "P6-BLEND-SUBMISSION-v1" if ENABLE_P6 else RUN_ID',
        "effective_run_id = RUN_ID",
    )
    deployment = deployment.replace("p6_blend_", "p8_blend_")
    deployment = replace_once(
        deployment,
        'manifest_path = OUTPUT_ROOT / (\n'
        '        "p8_blend_submission_manifest.json"\n'
        '        if ENABLE_P6\n'
        '        else "p8_blend_submission_manifest.json"\n'
        "    )",
        'manifest_path = OUTPUT_ROOT / "p8_blend_submission_manifest.json"',
    )
    deployment = deployment.replace(
        "P6_BLEND_SUBMISSION_START", "P8_BLEND_SUBMISSION_START"
    )
    deployment = deployment.replace(
        "P6_BLEND_MODELS_READY", "P8_BLEND_MODELS_READY"
    )
    deployment = deployment.replace(
        "P6_BLEND_SUBMISSION_COMPLETE", "P8_BLEND_SUBMISSION_COMPLETE"
    )
    deployment = deployment.replace(
        "Full six-checkpoint inference", "Full seven-backbone inference"
    )
    deployment = deployment.replace(
        "dicom_and_six_model_inference_minutes",
        "dicom_and_seven_model_inference_minutes",
    )
    assert "P4" not in deployment and "p4" not in deployment
    assert deployment.count("rsna-knee-p8-noisy-student-fold-") == 1
    assert deployment.count("P8-NOISY-STUDENT-FOLD-") == 1
    assert deployment.count("p8_blend_submission_manifest.json") == 1
    assert deployment.count("P8-BLEND-SUBMISSION-v1") == 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(common_prefix + "\n\n" + deployment)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
