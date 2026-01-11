from pydantic import BaseModel
from typing import Optional, List

class FinancialRow(BaseModel):
    period: Optional[str]
    revenue_actual: Optional[float]
    revenue_expected: Optional[float]
    eps_actual: Optional[float]
    eps_expected: Optional[float]
    forecast_actual: Optional[float]


class Financials(BaseModel):
    rows: List[FinancialRow]

    @classmethod
    def from_llm(cls, data):
        # Accept both list-of-dicts and dict with rows key
        if isinstance(data, dict) and 'rows' in data:
            data_in = data['rows']
        else:
            data_in = data
        if not isinstance(data_in, list):
            raise ValueError('LLM data must be a list of objects')
        rows = [FinancialRow(**r) for r in data_in]
        return cls(rows=rows)

