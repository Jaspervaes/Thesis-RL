"""Generate SimBank data — delegates to shared/generate_data.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from shared.generate_data import main; main()
