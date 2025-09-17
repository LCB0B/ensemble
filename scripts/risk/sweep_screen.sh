trap 'kill 0' SIGINT
## Age
python scripts/risk/screening.py model_type=age screening_age_lower=30 
python scripts/risk/screening.py model_type=age screening_age_lower=40
python scripts/risk/screening.py model_type=age screening_age_lower=50
python scripts/risk/screening.py model_type=age screening_age_lower=60
python scripts/risk/screening.py model_type=age screening_age_lower=70

## Risk
python scripts/risk/screening.py model_type=risk risk_threshold=0.00001
python scripts/risk/screening.py model_type=risk risk_threshold=0.0001
python scripts/risk/screening.py model_type=risk risk_threshold=0.001
python scripts/risk/screening.py model_type=risk risk_threshold=0.01
