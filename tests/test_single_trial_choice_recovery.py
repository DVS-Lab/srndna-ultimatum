"""Regression check for the 55-volume public series in sub-104 run 2."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from extract_single_trial_choice import recover_104


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


if __name__ == "__main__":
    unittest.main()
