"""Saved FaceNet classifier checks; these do NOT validate actual image inference."""
from pathlib import Path
import sys, unittest, os, warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import face_model as fm
class FacePipelineTests(unittest.TestCase):
    def test_threshold_never_clips_below_stranger(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertGreater(fm.choose_threshold([.9], [.99]), .99)
            self.assertGreater(fm.choose_threshold([], [1.0]), 1.0)
        with self.assertRaises(ValueError): fm.choose_threshold([.9], [float('nan')])
    def test_no_detected_face_is_denied(self):
        with patch.object(fm,'load_rgb',return_value=object()),patch.object(fm,'extract_features',side_effect=fm.NoFaceDetected('none')):
            self.assertEqual(fm.predict_face('mocked'),('unknown',0.0))
    def test_existing_backend_does_not_silently_change(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(fm.default_backend(),'facenet')
    def test_split_excludes_placide_from_calibration(self):
        calibration,regression=fm.split_unauthorized_images()
        self.assertEqual(len(regression),3)
        self.assertTrue(set(calibration).isdisjoint(regression))
        self.assertFalse(any(Path(p).name.lower().startswith('placide_') for p in calibration))
    def test_saved_embeddings_match_model(self):
        b=fm.load_bundle(); df=pd.read_csv(ROOT/'data/image_features.csv')
        x=df[fm.feature_columns(df)].to_numpy()
        self.assertEqual(x.shape,(63,512)); self.assertEqual(x.shape[1],b['n_features'])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(b['pipeline'].predict_proba(x).shape,(63,3))
    def test_names_align_after_lowercasing(self):
        b=fm.load_bundle(); voice=joblib.load(ROOT/'models/voice_model.pkl')
        self.assertEqual(sorted(c.lower() for c in b['classes']),sorted(voice['classes']))
    def test_saved_backend_used(self):
        b=fm.load_bundle(); df=pd.read_csv(ROOT/'data/image_features.csv'); x=df[fm.feature_columns(df)].iloc[0].to_numpy()
        with patch.object(fm,'load_rgb',return_value=object()),patch.object(fm,'extract_features',return_value=x) as extract:
            name,confidence=fm.predict_face('mocked_input')
        self.assertEqual(extract.call_args.args[1],b['backend'])
        scores=b['pipeline'].predict_proba(x[None,:])[0]
        self.assertAlmostEqual(confidence,float(scores.max()))
        self.assertEqual(name,'unknown' if confidence<b['threshold'] else b['classes'][scores.argmax()])
if __name__=='__main__': unittest.main()
