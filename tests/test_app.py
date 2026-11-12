"""Control-flow tests for Kate's app; actual face inference requires FaceNet."""
import contextlib, io, sys, unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import app
class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.users=pd.DataFrame({'name':['Gilbert'],'customer_id':['A180']})
    def test_face_denial_stops_later_steps(self):
        with patch.object(app,'predict_face',return_value=('unknown',.2)), patch.object(app,'predict_product') as p, patch.object(app,'verify_voice') as v, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(app.run_transaction('face','voice',self.users))
        p.assert_not_called(); v.assert_not_called()
    def test_voice_denial_hides_product(self):
        with patch.object(app,'predict_face',return_value=('Gilbert',.99)), patch.object(app,'predict_product',return_value='SECRET_PRODUCT'), patch.object(app,'verify_voice',return_value=(False,.6)) as v, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(app.run_transaction('face','voice',self.users))
        v.assert_called_once_with('voice','gilbert')
        self.assertNotIn('SECRET_PRODUCT',out.getvalue())
    def test_success_order(self):
        calls=[]
        with patch.object(app,'predict_face',side_effect=lambda p:(calls.append('face') or ('Gilbert',.99))), patch.object(app,'predict_product',side_effect=lambda cid:(calls.append(('product',cid)) or 'Clothing')), patch.object(app,'verify_voice',side_effect=lambda p,n:(calls.append(('voice',n)) or (True,.98))), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(app.run_transaction('face','voice',self.users))
        self.assertEqual(calls,['face',('product','A180'),('voice','gilbert')])
        self.assertIn('Clothing',out.getvalue())
    def test_no_filename_fallback_on_voice_error(self):
        with patch.object(app,'predict_face',return_value=('Gilbert',.99)), patch.object(app,'predict_product',return_value='SECRET_PRODUCT'), patch.object(app,'verify_voice',side_effect=ImportError('SECRET_PRODUCT')), contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(RuntimeError): app.run_transaction('face','Gilbert.wav',self.users)
        self.assertNotIn('SECRET_PRODUCT',out.getvalue())
    def test_scenario_files_exist(self):
        for member in ['Amazing','Gilbert','Kate']:
            for scenario in 'abc':
                for path in app.scenario_paths(scenario,member): self.assertTrue(Path(path).is_file(),path)
    def test_main_fails_closed_on_missing_model(self):
        with patch.object(app,'load_users',return_value=self.users), patch.object(app,'run_transaction',side_effect=RuntimeError('SECRET_PRODUCT')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(app.main(['--face','x','--voice','y']),2)
        self.assertNotIn('SECRET_PRODUCT',out.getvalue())
    def test_all_scenarios_require_expected_outcomes(self):
        for outputs,code in [([True,False,False],0),([False,False,False],1),([True,True,False],1)]:
            with patch.object(app,'load_users',return_value=self.users), patch.object(app,'run_scenario',side_effect=outputs), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(['--scenario','all']),code)
    def test_no_prediction_stops_voice(self):
        with patch.object(app,'predict_face',return_value=('Gilbert',.99)), patch.object(app,'predict_product',return_value=None), patch.object(app,'verify_voice') as v, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError): app.run_transaction('face','voice',self.users)
        v.assert_not_called()
if __name__=='__main__': unittest.main()
