from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from veg_recovery.contracts import ReconstructionRequest, ReconstructionPayload, validate_request
from veg_recovery.data import read_dataset, summarize_frame, validate_frame, TEST_COLUMNS
from veg_recovery.features import FeatureState, build_features, fit_feature_state, infer_source_labels
from veg_recovery.models.baselines import BaselineModel, BASELINE_METHODS
from veg_recovery.models.bundle import save_baseline_bundle, save_trained_bundle, load_bundle
from veg_recovery.models.calibration import select_blend, fit_uncertainty, evaluate_uncertainty
from veg_recovery.validation import (MaskSpec, MASKED_COLUMNS, apply_mask, generate_folds, split_fold,
                                     context_diagnostics, gap_score)
from veg_recovery.inference import load_reconstructor
from veg_recovery.cli.batch import validate_submission_file, validate_submission, write_submission
from veg_recovery.anomalies.baseline import detect_anomalies, fit_harmonization, harmonize_values, reconstruct_product_series

ROOT = Path(__file__).resolve().parents[2]


class ConstantPredictor:
    """A test double, constructed without any fit/training."""
    def __init__(self,value): self.value=value
    def predict(self,features): return np.full(len(features),self.value)


class ConstantSource:
    classes_=np.array(['s2','landsat','modis','unknown'])
    def predict_proba(self,features): return np.tile([.7,.2,.1,0.],(len(features),1))


def fixture(polygons=3, years=(2022, 2023, 2024), n=10):
    records = []
    for p in range(polygons):
        for year in years:
            for i, date in enumerate(pd.date_range(f"{year}-06-01", periods=n, freq="5D")):
                row = {c: np.nan for c in TEST_COLUMNS}
                y = .2 + .02 * i + p * .01
                row.update(anon_polygon_id=f"p{p}", crop_type="crop", date=date,
                           primary_ndvi=y, s2_ndvi=y, s2_evi=.3, s2_ndwi=.1,
                           era5_temp_c=20., era5_precip_mm=1., year=float(year),
                           doy=float(date.dayofyear), ndvi_climatology_mean=y,
                           ndvi_climatology_std=.1, n_reference_years=3., is_synthetic_gap=False)
                records.append(row)
    return pd.DataFrame(records)


class DataTests(unittest.TestCase):
    def test_current_controls(self):
        train = read_dataset(ROOT / "data/train_dataset.csv", "train", strict_current=True)
        test = read_dataset(ROOT / "data/test_data.csv", "test", strict_current=True)
        self.assertEqual(len(set(train.anon_polygon_id) & set(test.anon_polygon_id)), 39)
        self.assertEqual(int(test.loc[test.is_synthetic_gap, 'anon_polygon_id'].isin(train.anon_polygon_id).sum()), 464)
        self.assertEqual(summarize_frame(train)['finite_targets'], 30520)
        self.assertTrue(test.loc[test.is_synthetic_gap, list(set(MASKED_COLUMNS) & set(test))].isna().all().all())

    def test_duplicate_schema_nonfinite(self):
        data = fixture()
        with self.assertRaises(ValueError): validate_frame(pd.concat([data, data.iloc[:1]]))
        with self.assertRaises(ValueError): validate_frame(data.drop(columns='crop_type'))
        data.loc[0, 'primary_ndvi'] = np.inf
        state = fit_feature_state(data)
        self.assertEqual(state.fit_count, len(data) - 1)

    def test_exact_mask_and_no_mutation(self):
        data = fixture(); data['status'] = 'good'; data['ndvi_zscore'] = 1.
        original = data.copy(deep=True); keys = data.iloc[[2, 4]][['anon_polygon_id', 'date']]
        masked = apply_mask(data, keys)
        self.assertTrue(masked.loc[[2,4], list(set(MASKED_COLUMNS)&set(data))].isna().all().all())
        assert_frame_equal(original, data)
        self.assertEqual(masked.loc[2, 'crop_type'], 'crop')
        inferred = MaskSpec.from_test(masked)
        self.assertIn('status', inferred.columns)

    def test_hierarchy(self):
        data = fixture().iloc[:5].copy()
        data['primary_ndvi'] = [.2, .3, .4, .5, np.inf]
        data['s2_ndvi'] = [.2, .1, np.nan, .1, np.inf]
        data['landsat_ndvi'] = [.2, .3, np.nan, .2, np.inf]
        data['modis_ndvi'] = [.2, .3, .4, .3, np.inf]
        self.assertEqual(infer_source_labels(data).tolist(), ['s2','landsat','modis','unknown','unknown'])
        self.assertEqual(gap_score(.06), 12.)


class LeakageAndFoldTests(unittest.TestCase):
    def test_sentinel_all_dynamic(self):
        data = fixture(); keys = data.iloc[[5,6]][['anon_polygon_id','date']]
        state = fit_feature_state(data, excluded_keys=keys)
        first = build_features(apply_mask(data, keys), keys, state)
        poisoned = data.copy(); poisoned.loc[[5,6], list(set(MASKED_COLUMNS)&set(data))] = 1e9
        second = build_features(apply_mask(poisoned, keys), keys, state)
        assert_frame_equal(first, second, check_exact=True)
        assert_frame_equal(first, build_features(poisoned, keys, state), check_exact=True)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            build_features(data, keys, fit_feature_state(data))

    def test_grouped_exclusion(self):
        data = fixture(); held = data.anon_polygon_id.eq('p0')
        a = fit_feature_state(data, excluded_polygons=['p0'])
        changed = data.copy(); changed.loc[held,'primary_ndvi'] += 1e9
        b = fit_feature_state(changed, excluded_polygons=['p0'])
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertNotIn('p0', a.seen_polygons)

    def test_same_date_excludes_polygon(self):
        data = fixture(years=(2024,)); keys = data.iloc[[3]][['anon_polygon_id','date']]
        f = build_features(data, keys, FeatureState())
        self.assertEqual(f.iloc[0].same_date_target_count, 2)
        self.assertAlmostEqual(f.iloc[0].same_date_target_median, .275)
        self.assertAlmostEqual(f.iloc[0].days_left_1, 5.)
        details = context_diagnostics(data, keys)
        self.assertEqual(details.iloc[0].left_days, 5.)
        self.assertEqual(details.iloc[0].right_days, 5.)

    def test_deterministic_folds_past_and_hard(self):
        data = fixture(n=14)
        folds = generate_folds(data, seeds=(17,29,43,71,101), n_splits=3, max_gaps_per_split=9)
        assert_frame_equal(folds, generate_folds(data.sample(frac=1, random_state=9), seeds=(17,29,43,71,101), n_splits=3, max_gaps_per_split=9))
        self.assertEqual(set(folds['mode']), set('ABCD'))
        self.assertEqual(folds.loc[folds['mode'].eq('A'), 'repeat'].nunique(), 5)
        for mode, repeat, fold in folds[['mode','repeat','fold']].drop_duplicates().itertuples(index=False, name=None):
            part = split_fold(data, folds, mode, repeat, fold)
            state = fit_feature_state(part.fit_frame)
            features = build_features(part.context_frame, part.gap_keys, state)
            if mode in ['B','D']:
                self.assertFalse(set(part.targets.anon_polygon_id) & set(state.seen_polygons))
            if mode == 'C': self.assertLess(part.fit_frame.date.max(), pd.Timestamp('2024-01-01'))
            if mode == 'D':
                self.assertTrue(features.days_left_1.gt(15).all())
                self.assertTrue(features.days_right_1.isna().all())
                self.assertTrue(features.gap_run_length.between(2,4).all())
                for p in part.targets.anon_polygon_id.unique():
                    self.assertEqual(np.isfinite(part.context_frame.loc[part.context_frame.anon_polygon_id.eq(p),'primary_ndvi']).sum(), 1)


class BaselineTests(unittest.TestCase):
    def test_interior_edge_length4_empty(self):
        data = fixture(polygons=1, years=(2024,), n=8)
        keys = data.iloc[[2,3,4,5]][['anon_polygon_id','date']]
        features = build_features(data, keys, FeatureState())
        self.assertTrue(features.gap_run_length.eq(4).all())
        linear = BaselineModel('linear').predict_from_features(features)
        np.testing.assert_allclose(linear.primary_ndvi_pred, [.24,.26,.28,.30])
        mean = BaselineModel('mean_neighbors').predict_from_features(features)
        np.testing.assert_allclose(mean.primary_ndvi_pred, [.27]*4)
        for keys in [data.iloc[[0]][['anon_polygon_id','date']], data[['anon_polygon_id','date']]]:
            for method in BASELINE_METHODS:
                pred = BaselineModel(method).predict(data, keys)
                self.assertTrue(np.isfinite(pred.primary_ndvi_pred).all())

    def test_raw_target_not_clipped(self):
        data = fixture(polygons=1, years=(2024,), n=3)
        data.loc[0, 'primary_ndvi'] = -2.
        features = build_features(data, data.iloc[[1]][['anon_polygon_id','date']], FeatureState())
        self.assertEqual(features.iloc[0].target_left_1, -2.)
        self.assertEqual(features.iloc[0].target_left_1_invalid_flag, 1)
        self.assertAlmostEqual(BaselineModel().predict_from_features(features).iloc[0].primary_ndvi_pred, -.88)

    def test_blend_convex_and_no_worse(self):
        oof = pd.DataFrame({'mode':list('ABCD')*3,'primary_ndvi':np.arange(12)/20})
        oof['pred_hgb'] = oof.primary_ndvi + .1
        oof['pred_extra_trees'] = oof.primary_ndvi - .1
        weights, _, pred, _ = select_blend(oof, ['hgb','extra_trees'])
        self.assertAlmostEqual(sum(weights.values()), 1.)
        self.assertTrue(all(w>=0 for w in weights.values()))
        np.testing.assert_allclose(pred, oof.primary_ndvi, atol=1e-6)


class BundleAndCLITests(unittest.TestCase):
    def test_verified_trained_path_with_test_doubles(self):
        from veg_recovery.features import source_feature_columns
        data=fixture(polygons=1,years=(2024,),n=5)
        keys=data.iloc[[2]][['anon_polygon_id','date']]
        data=apply_mask(data,keys); features=build_features(data,keys,FeatureState())
        estimators={'regressors':{'hgb':ConstantPredictor(.3),'extra_trees':ConstantPredictor(.4)},
                    'source_classifier':ConstantSource(),'source_columns':source_feature_columns(features)}
        with tempfile.TemporaryDirectory() as d:
            save_trained_bundle(d,FeatureState(),{'weights':{'hgb':.5,'extra_trees':.5},'feature_columns':list(features)},estimators)
            with self.assertRaisesRegex(ValueError,'trusted'): load_bundle(d)
            result=load_reconstructor(d,trusted=True).predict(ReconstructionRequest(data,data.is_synthetic_gap,'competition'))
            self.assertAlmostEqual(result.predictions.iloc[0].primary_ndvi_pred,.35)
            self.assertAlmostEqual(result.diagnostics.iloc[0].p_s2,.7)
            self.assertTrue(np.isfinite(result.predictions.lower).all())

    def test_bundle_manifest_hash_schema(self):
        with tempfile.TemporaryDirectory() as d:
            save_baseline_bundle(d, FeatureState())
            self.assertEqual(load_bundle(d).manifest.schema_version, '1.0')
            path = Path(d)/'config.json'; path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'SHA256'): load_bundle(d)
            manifest = Path(d)/'manifest.json'; content = json.loads(manifest.read_text()); content['schema_version']='99'
            manifest.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, 'schema_version'): load_bundle(d)

    def test_strict_submission_and_cli_api_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=fixture(polygons=1,years=(2024,),n=8)
            data=apply_mask(data,data.iloc[[2,4]][['anon_polygon_id','date']])
            data[TEST_COLUMNS].to_csv(root/'input.csv',index=False,date_format='%Y-%m-%d')
            save_baseline_bundle(root/'bundle', FeatureState())
            loaded=read_dataset(root/'input.csv','test')
            request=ReconstructionRequest(loaded,loaded.is_synthetic_gap,'competition')
            model=load_reconstructor(root/'bundle'); first=model.predict(request); second=model.predict(request)
            assert_frame_equal(first.predictions,second.predictions)
            payload=ReconstructionPayload.from_result(first)
            self.assertEqual(len(payload.predictions),2); self.assertNotIn('NaN',payload.model_dump_json())
            product=reconstruct_product_series(loaded,first)
            visible=~loaded.is_synthetic_gap
            np.testing.assert_array_equal(product.loc[visible,'primary_ndvi_reconstructed'],loaded.loc[visible,'primary_ndvi'])
            self.assertTrue(product.loc[loaded.is_synthetic_gap,'confidence'].between(0,1).all())
            command=[sys.executable,'-m','veg_recovery.cli.batch','--input',str(root/'input.csv'),'--bundle',str(root/'bundle'),
                     '--output',str(root/'submission.csv'),'--diagnostics',str(root/'diag.csv'),'--expected-count','2']
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            exported=validate_submission_file(root/'submission.csv',loaded.loc[loaded.is_synthetic_gap,['anon_polygon_id','date']])
            np.testing.assert_allclose(exported.primary_ndvi_pred,first.predictions.primary_ndvi_pred,atol=1e-15)
            bad=exported.copy(); bad['extra']=1
            with self.assertRaises(ValueError): validate_submission(bad,exported)
            for value in [np.nan,np.inf,'not-a-number']:
                bad=exported.copy(); bad['primary_ndvi_pred']=value
                with self.assertRaises(ValueError): validate_submission(bad,exported)
            with self.assertRaises(ValueError): validate_submission(exported.iloc[::-1],exported)
            (root/'submission.csv').write_bytes(b'\xff')
            with self.assertRaises(UnicodeError): validate_submission_file(root/'submission.csv',exported)
            run=subprocess.run(command[:-1]+['999'],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)

    def test_request_rejects_misaligned_mask_and_empty(self):
        data=fixture(); mask=pd.Series(False,index=data.index)
        with self.assertRaises(ValueError): validate_request(ReconstructionRequest(data,mask.iloc[::-1],'web'))
        with tempfile.TemporaryDirectory() as d:
            save_baseline_bundle(d,FeatureState())
            result=load_reconstructor(d).predict(ReconstructionRequest(data,mask,'web'))
            self.assertEqual(len(result.predictions),0)


class AnomalyTests(unittest.TestCase):
    def test_previous_years_negative_persistent_only(self):
        data=fixture(polygons=1,years=(2021,2022,2023,2024),n=4)
        data['ndvi_harmonized']=.6
        data.loc[data.date.dt.year.eq(2024),'ndvi_harmonized']=.2
        result=detect_anomalies(data)
        self.assertEqual(len(result.events),1)
        self.assertTrue((result.points.loc[result.points.date.dt.year.lt(2024),'reference_status']=='insufficient_reference_years').all())
        self.assertEqual(result.events.iloc[0].n_points,4)
        data.loc[data.date.dt.year.eq(2024),'ndvi_harmonized']=.9
        self.assertTrue(detect_anomalies(data).events.empty)

    def test_harmonization_separate(self):
        x=np.linspace(.1,.8,100); frame=pd.DataFrame({'s2_ndvi':x,'landsat_ndvi':(x-.05)/.8})
        original=frame.copy(); calibration=fit_harmonization(frame)
        transformed,_=harmonize_values(np.array([.5]),pd.DataFrame({'p_landsat':[1.]}),calibration)
        self.assertAlmostEqual(transformed[0],.45)
        assert_frame_equal(original,frame)


class OfflineTrainingContractTests(unittest.TestCase):
    def test_requires_explicit_training_flag(self):
        run=subprocess.run([sys.executable,'-m','veg_recovery.models.training'],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('training disabled',run.stderr)

    def test_pseudo_training_sentinel_without_fitting_models(self):
        from veg_recovery.models.training import prepare_pseudo_training
        data=fixture(n=14)
        with patch('sklearn.ensemble.HistGradientBoostingRegressor.fit',side_effect=AssertionError('Local ML training forbidden')):
            features,truth=prepare_pseudo_training(data,seed=17,max_samples=15)
        self.assertTrue(features.primary_ndvi_raw.isna().all())
        self.assertTrue(np.isfinite(truth.primary_ndvi).all())
        self.assertTrue(features[['anon_polygon_id','date']].equals(truth[['anon_polygon_id','date']]))

    def test_gpu_factory_contract_without_fitting(self):
        from veg_recovery.models.estimators import make_regressor
        features=pd.DataFrame({'anon_polygon_id':['p1'],'date':[pd.Timestamp('2024-01-01')],
                                   'numeric_feature':[1.]})
        catboost=make_regressor('catboost',features,device='gpu',n_jobs=2)
        self.assertEqual(catboost.device,'gpu')
        with self.assertRaisesRegex(ValueError,'does not support GPU'):
            make_regressor('hgb',features,device='gpu')

        notebook=json.loads(Path('artifacts/ml/kaggle/train_ndvi.ipynb').read_text())
        source=''.join(''.join(cell.get('source',[])) for cell in notebook['cells'])
        self.assertIn('--device", "gpu", "--models", "catboost',source)
        self.assertIn('RUN_TUNING = False',source)


if __name__=='__main__': unittest.main()
