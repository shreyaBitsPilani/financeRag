# guardrails.py

import re
"""
Very naive guardrails to filter out harmful queries (input filtering)
and to detect or adjust misleading/hallucinated outputs (output filtering).
In practice, you'd want something more robust, possibly a policy-based approach.
"""

# Simple keywords for demonstration
HARMFUL_KEYWORDS = ["kill", "attack", "terrorist", "bomb", "hate", "abuse"]
PROHIBITED_TOPICS = ["violent", "illegal"]
FINANCE_KEYWORDS = {
    "income_statement": [
        # Revenue and Sales
        "revenue", 
        "sales",
        "top line",
        "turnover",

        # Income / Profit
        "income",
        "profit",
        "net profit",
        "net earnings",
        "net income",
        "loss",

        # Operating Income / Profit
        "operating income",
        "operating profit",
        "operating margin",

        # Gross Margin / Gross Profit
        "gross margin",
        "gross profit",

        # Costs and Expenses
        "cost",
        "expense",
        "cost of goods sold",             # COGS
        "cogs",                           # abbreviation

        # Interest and Taxes
        "interest expense",
        "tax expense",
        "non-operating income",
        "other expenses",

        # EBIT / EBITDA
        "ebit",                           # Earnings Before Interest & Taxes
        "earnings before interest and taxes",
        "ebitda",                         # Earnings Before Interest, Taxes, Depreciation & Amortization
        "earnings before interest, taxes, depreciation, and amortization",

        # Earnings per Share
        "eps",                            # Earnings Per Share
        "earnings per share",
        "diluted eps",
        "diluted earnings per share"
    ],
    "balance_sheet": [
        # Assets
        "assets",
        "current assets",
        "non-current assets",
        "long-term assets",
        "cash",
        "cash equivalents",
        "inventory",
        "receivables",
        "accounts receivable",
        "marketable securities",
        "property, plant & equipment",    # PP&E
        "pp&e",
        "intangible assets",
        "goodwill",

        # Liabilities
        "liabilities",
        "current liabilities",
        "short-term liabilities",
        "long-term liabilities",
        "non-current liabilities",
        "accounts payable",
        "debt",
        "short-term debt",
        "long-term debt",

        # Equity
        "equity",
        "shareholders’ equity",
        "share capital",
        "preferred stock",
        "common stock",
        "treasury stock",
        "retained earnings",
        "accumulated profit"
    ],
    "cash_flow": [
        # Cash Flow Overview
        "cash flow",
        "cashflow",
        "cfo",                            # Cash From Operations
        "cash from operations",
        "operating cash flow",
        "investing activities",
        "cash used in investing",
        "financing activities",
        "cash from financing",

        # Free Cash Flow / Capital Expenditures
        "free cash flow",
        "fcf",                            # free cash flow (abbreviated)
        "capital expenditures",
        "capex",

        # Changes in Cash
        "net change in cash",
        "net increase in cash",
        "net decrease in cash",

        # Financing Details
        "dividends paid",
        "stock repurchase",
        "issuance of debt",
        "repayment of debt",
        "interest received",
        "interest paid"
    ]
}


def is_financial_question(query, keywords_dict=FINANCE_KEYWORDS):
    """
    Returns True if the query contains at least one financial keyword 
    from any of the categories in 'keywords_dict'. Otherwise False.
    Case-insensitive match on whole words only.
    """
    query_lower = query.lower()
    
    # Simple approach: For each category in the dictionary, 
    # check if any keyword is present in the query. 
    # You might do more advanced NLP as well.
    for category, keywords in keywords_dict.items():
        for kw in keywords:
            # Naive whole-word match using regex word boundaries
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, query_lower):
                return True
    return False

def reject_harmful_query(query):
    """
    Reject queries containing harmful keywords or topics.
    Return True if the query is rejected, otherwise False.
    """
    lower_query = query.lower()
    for kw in HARMFUL_KEYWORDS + PROHIBITED_TOPICS:
        if kw in lower_query:
            return True
    return False

def filter_misleading_output(text):
    """
    Naive approach to detect if the text is purely hallucinated or contradictory.
    Real solutions might use specialized classifiers or logic.
    
    For demonstration, we just check if the text contains suspicious patterns 
    like "undefined" or "NaN" or "???", etc.
    """
    suspicious_markers = ["???", "undefined", "NaN", "no data"]
    lower_text = text.lower()
    for marker in suspicious_markers:
        if marker in lower_text:
            # Return a sanitized message or an empty string
            return "Output flagged as potentially misleading."
    return text

if __name__ == "__main__":
    # Demo
    queries = [
        "What is the revenue for 2023?",
        "I want to bomb the building"
    ]
    for q in queries:
        if reject_harmful_query(q):
            print(f"Rejected query: {q}")
        else:
            print(f"Accepted query: {q}")
    
    outputs = [
        "The company's revenue is ??? for 2023",
        "Revenue: 50000"
    ]
    for o in outputs:
        print("Filtered output:", filter_misleading_output(o))
