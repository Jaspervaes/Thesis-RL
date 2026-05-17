"""Generate SimBank data — delegates to shared/generate_data.py."""
import sys, os
_r = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _r)
sys.path.insert(0, os.path.join(_r, "methods"))
os.chdir(_r)
from shared.generate_data import main; main()
