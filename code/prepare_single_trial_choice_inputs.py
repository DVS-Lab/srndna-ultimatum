"""Regenerate the release fingerprint table and fixed frontal-pole mask from their source files."""
import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from extract_figure4_conditions import read_rows


def prepare(repo, dataset_git, atlas, out):
    snap = "95ff4f58ae90d37825c9bb2bf4341db44704146b"
    def git(*args):
        return subprocess.check_output(["git","-C",str(dataset_git),*args],text=True).strip()
    if git("rev-parse","2.2.0^{commit}") != snap:
        raise ValueError("OpenNeuro tag differs from the verified 2.2.0 snapshot")
    source = repo/"results/palm_age_main/v1/preparation/full-rt__nppi-dmn"
    out.mkdir(parents=True,exist_ok=True)
    fingerprints = []
    for member in read_rows(source/"participants.tsv"):
        subject = member["subject"]
        for run in (1,2):
            prefix = f"{subject}_task-ultimatum_run-{run:02d}"
            image = f"derivatives/single_trials/{subject}/{prefix}_singletrial-Act.nii.gz"
            key = Path(git("show",f"{snap}:{image}")).name
            match = re.fullmatch(r"(MD5|SHA256)E-s(\d+)--([a-f0-9]+)\.nii\.gz",key)
            if match is None:
                raise ValueError(f"Unexpected git-annex key for {image}")
            event = git("rev-parse",f"{snap}:{subject}/func/{prefix}_events.tsv")
            fingerprints.append([subject,run,image,*match.groups(),event])
    if len(fingerprints) != 94:
        raise ValueError("Unexpected imaging roster")
    with (out/"openneuro_2.2.0_inputs.tsv").open("w",newline="") as stream:
        writer = csv.writer(stream,delimiter="\t",lineterminator="\n")
        writer.writerow(["subject","run","image","algorithm","bytes","digest","events_git_blob"])
        writer.writerows(fingerprints)
    if hashlib.sha256(atlas.read_bytes()).hexdigest() != "7397ffdbae7559e0f0aa6237f998dc0f6aca3f9db9009f6ffa637c0c62b686ca":
        raise ValueError("Expected the Harvard–Oxford cortical 2-mm max-probability atlas at 25%")
    ref = nib.load(source/"mask.nii.gz")
    labels = resample_from_to(nib.load(atlas),ref,order=0).get_fdata()
    indices = np.indices(ref.shape).reshape(3,-1).T
    right = (nib.affines.apply_affine(ref.affine,indices)[:,0] > 0).reshape(ref.shape)
    roi = (labels==1) & right & (ref.get_fdata()>0)
    if roi.sum() != 2018:
        raise ValueError("Right frontal pole coverage differs from the analysis definition")
    nib.save(nib.Nifti1Image(roi.astype("uint8"),ref.affine),out/"right_frontal_pole.nii.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset-git",type=Path,required=True)
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    prepare(a.repo,a.dataset_git,a.atlas,a.output)
