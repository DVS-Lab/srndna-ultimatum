"""Regression check for the 55-volume public series in sub-104 run 2."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from extract_single_trial_choice import VERSION, archived_trial_ids, recover_104, verified_packed_104


class TrialRecovery(unittest.TestCase):
    def test_only_verified_trials_can_be_reintroduced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/"sub-104/LSS-images_task-ultimatum_model-01_type-act_run-02"
            evs = root/"EVfiles/sub-104/SingleTrialEVs/run02"
            folder.mkdir(parents=True)
            evs.mkdir(parents=True)
            events = root/"events.tsv"
            rng = np.random.default_rng(13)
            data = rng.normal(size=(4,3,2,72)).astype(np.float32)
            indices = sorted(rng.choice(72,55,replace=False))
            released = nib.Nifti1Image(data[...,indices],np.eye(4))
            with events.open("w",newline="") as out:
                writer = csv.writer(out,delimiter="\t")
                writer.writerow(["trial_type","onset","duration"])
                for j in range(1,73):
                    writer.writerow(["event_accept_ingroup",j,1])
                    (evs/f"trialmodel-{j}_estimage-single.tsv").write_text(f"{j} 1 1\n")
                    nib.save(nib.Nifti1Image(data[...,j-1],np.eye(4)),folder/f"zstat_trial-{j:02}.nii.gz")
            img, ids = recover_104(released,events,[root])
            np.testing.assert_array_equal(img.get_fdata(),data)
            self.assertEqual(ids,list(range(1,73)))
            # A missing image outside the public set permits only the 55 exact matches.
            absent = next(j for j in range(72) if j not in indices)
            (folder/f"zstat_trial-{absent+1:02}.nii.gz").unlink()
            img, ids = recover_104(released,events,[root])
            np.testing.assert_array_equal(img.get_fdata(),released.get_fdata())
            self.assertEqual(ids,[i+1 for i in indices])
            # A wrong onset inside the public set makes the sequence unusable.
            (evs/f"trialmodel-{indices[0]+1}_estimage-single.tsv").write_text("0 1 1\n")
            self.assertEqual(recover_104(released,events,[root]),(None,[]))

    def test_roi_fingerprints_identify_interleaved_trials_and_validate_a_full_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"masks").mkdir()
            (root/VERSION).mkdir(parents=True)
            data = np.random.default_rng(12).normal(size=(2,2,2,72)).astype(np.float32)
            names = ("VMPFC","aINS","dACC","VS")
            for i,name in enumerate(names):
                mask = np.zeros((2,2,2),dtype=np.uint8)
                mask.flat[i] = 1
                nib.save(nib.Nifti1Image(mask,np.eye(4)),root/f"masks/roi-{name}.nii.gz")
            with (root/VERSION/"sub104_roi_reference.tsv").open("w") as stream:
                writer = csv.writer(stream,delimiter="\t")
                writer.writerow(["run","trial",*names])
                for j in range(72):
                    writer.writerow([2,j+1,*data[...,j].ravel()[:4]])
            ids = list(range(1,54)) + [59,67]
            public = nib.Nifti1Image(data[...,np.array(ids)-1],np.eye(4))
            self.assertEqual(archived_trial_ids(public,root,2)[0],ids)
            directory = root/"derivatives/single_trials/sub-104"
            directory.mkdir(parents=True)
            packed = directory/"sub-104_task-ultimatum_run-02_singletrial-Act.nii.gz"
            nib.save(nib.Nifti1Image(data,np.eye(4)),packed)
            workroots = [root/"derivatives/fsl"]
            self.assertIsNotNone(verified_packed_104(public,root,workroots,ids))
            # A candidate that matches all 55 published volumes must also validate
            # the reference fingerprint for every additional volume.
            data[...,53] = data[...,54]
            nib.save(nib.Nifti1Image(data,np.eye(4)),packed)
            self.assertIsNone(verified_packed_104(public,root,workroots,ids))


if __name__ == "__main__":
    unittest.main()
