"""Проверки реальных рисков: утечки, самоподглядывание и формат контрольных строк."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ndvi.data import VALUES, hide, query_only
from ndvi.features import build_features
from ndvi.pipeline import validate_submission, folds_for


def fixture():
    dates=pd.date_range("2020-04-01",periods=7,freq="4D")
    d=pd.DataFrame({"row_id":range(14),"anon_polygon_id":["P1"]*7+["P2"]*7,
                    "date":list(dates)*2,"crop_type":"зерновые"})
    d["year"]=d.date.dt.year;d["doy"]=d.date.dt.dayofyear
    d["day"]=(d.date-pd.Timestamp("2000-01-01")).dt.days
    for i,c in enumerate(VALUES):d[c]=np.linspace(.3,.8,len(d))+i*.005
    d["is_synthetic_gap"]=False
    return d.set_index("row_id",drop=False)


class IntegrityTests(unittest.TestCase):
    def test_unhidden_query_rejected(self):
        d=fixture()
        with self.assertRaises(ValueError):build_features(d.loc[[2]],d)

    def test_hidden_values_invariant(self):
        d=fixture();ids=[2,3]
        a=build_features(d.loc[ids],hide(d,ids))
        poisoned=d.copy();poisoned.loc[ids,VALUES]=99999
        b=build_features(poisoned.loc[ids],hide(poisoned,ids))
        pd.testing.assert_frame_equal(a,b)

    def test_neighbor_skips_entire_mask(self):
        d=fixture();ids=[2,3]
        x=build_features(d.loc[ids],hide(d,ids))
        self.assertAlmostEqual(x.loc[2,"primary_prev1"],d.loc[1,"primary_ndvi"],places=6)
        self.assertAlmostEqual(x.loc[2,"primary_next1"],d.loc[4,"primary_ndvi"],places=6)
        self.assertEqual(x.loc[2,"primary_next1_days"],8)

    def test_no_target_derived_columns(self):
        q=query_only(fixture())
        self.assertFalse(set(VALUES)&set(q))
        self.assertNotIn("status",q)

    def test_submission_only_gaps(self):
        d=fixture();d.loc[[2,10],"is_synthetic_gap"]=True
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"submission.csv"
            sub=validate_submission(d,[.4,.6],p)
            self.assertEqual(list(sub.columns),["anon_polygon_id","date","primary_ndvi_pred"])
            self.assertEqual(list(sub.anon_polygon_id),["P1","P2"])
            with self.assertRaises(ValueError):validate_submission(d,[np.nan,.2],p)

    def test_group_assignment_stable(self):
        d=fixture()
        self.assertEqual(folds_for(d),folds_for(d.sample(frac=1,random_state=9)))


if __name__=="__main__":unittest.main()
