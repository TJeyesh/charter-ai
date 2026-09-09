from fastapi import APIRouter, HTTPException
from datetime import datetime

from src.api.serializers import AnalyzeVoyageRequest, AnalyzeVoyageResponse
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.data.mock_db import get_mock_vessel_db, get_mock_port_info, require_sih_demo_mode

router = APIRouter(prefix="/analyze-voyage", tags=["Analysis"])

@router.post("", response_model=AnalyzeVoyageResponse)
async def analyze_voyage(request: AnalyzeVoyageRequest):
    require_sih_demo_mode()
    
    engine = DecisionEngine()
    
    origin_info = get_mock_port_info(request.origin)
    dest_info = get_mock_port_info(request.destination)
    
    expected_loading = datetime.now()
    required_delivery = datetime.combine(request.required_delivery_date, datetime.min.time())
    
    inputs = DecisionEngineInputs(
        cargo_type=request.cargo_type,
        cargo_quantity_t=request.cargo_quantity,
        origin_port_id=request.origin,
        destination_port_id=request.destination,
        expected_loading_date=expected_loading,
        required_delivery_date=required_delivery,
        number_of_voyages=request.number_of_voyages,
        contract_preference=request.contract_preference,
        vessel_specs_db=get_mock_vessel_db(),
        origin_port_info=origin_info,
        destination_port_info=dest_info
    )
    
    result = engine.evaluate(inputs)
    
    if result["status"] == "ERROR":
        raise HTTPException(status_code=400, detail=result)
        
    return result
