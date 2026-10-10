"""Run focused frozen-result, generator-sanity and final-document checks without long simulations."""
from pathlib import Path
import json,runpy,unittest,contextlib,io,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    counts={};start=time.time()
    for rel in ['tests/test_synthetic_identifiability_benchmark_v6.py','independent_generator_R_v6_1/tests/test_v6_robustness_generator.py']:
        ns=runpy.run_path(str(ROOT/rel));count=0
        for name,fn in ns.items():
            if name.startswith('test_') and callable(fn):fn();count+=1
        counts[rel]=count
    suite=unittest.TestSuite()
    for rel in ['tests/test_v6_2_additions.py','tests/test_v7_final.py']:
        ns=runpy.run_path(str(ROOT/rel))
        for obj in ns.values():
            if isinstance(obj,type) and issubclass(obj,unittest.TestCase):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    total=sum(counts.values())+result.testsRun
    summary={'generator_function_tests':counts,'unittest_checks':result.testsRun,'total_passed':total,
        'long_4000_and_9000_world_runs_repeated':False,'new_analyses_regenerated':True,'elapsed_seconds':round(time.time()-start,2),
        'scope':'Frozen benchmark outputs, generator sanity, completion witnesses, decisions, references, prose display citations and native equations.'}
    (ROOT/'results/v7_final_additions/validation_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
