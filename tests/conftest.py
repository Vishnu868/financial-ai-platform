"""
Pytest fixtures shared across all test files.
Provides pre-built sample texts and service instances.
"""

import pytest
import sys
import os

# Ensure app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.extraction_service import InvoiceExtractionService
from app.services.validator import ExtractionValidator
from app.services.bank_extractor import BankStatementExtractor


@pytest.fixture
def extractor():
    return InvoiceExtractionService()


@pytest.fixture
def validator():
    return ExtractionValidator()


@pytest.fixture
def bank_extractor():
    return BankStatementExtractor()


@pytest.fixture
def sample_amazon_text():
    return """
    Tax Invoice
    Amazon.in
    Invoice Number: IN-12345678
    Order ID: 408-1234567-8901234
    Order Date: 15/01/2025

    Sold by: TechStore India Private Limited
    GSTIN: 29AABCT1234F1Z5
    PAN: AABCT1234F

    Ship To: Nani
    Delivery Address: Hyderabad, Telangana 500001
    Phone: +91 9876543210

    Description                    Qty    Price
    USB-C Cable 1m                  2    ₹299.00
    Phone Case - Transparent        1    ₹499.00

    Subtotal: ₹1,097.00
    CGST @9%: ₹98.73
    SGST @9%: ₹98.73
    Total Tax: ₹197.46
    Grand Total: ₹1,294.46

    Payment Method: Amazon Pay
    Place of Supply: Telangana
    """


@pytest.fixture
def sample_flipkart_text():
    return """
    Flipkart Internet Private Limited
    Tax Invoice
    Invoice No: FKI-2025-0001234
    Order ID: OD123456789012

    GSTIN: 29AADCF0310E1Z6
    Order Date: 20 Jan 2025

    Sold by: Flipkart Seller Hub

    Product: Realme Buds Q2 Earbuds
    Qty: 1
    MRP: ₹1,999.00
    Discount: ₹500.00
    Subtotal: ₹1,499.00
    CGST @9%: ₹134.91
    SGST @9%: ₹134.91
    Total: ₹1,768.82

    Payment: Credit Card
    """


@pytest.fixture
def sample_swiggy_text():
    return """
    Swiggy
    Bundl Technologies Private Limited

    Bill No: SWG-2025-98765
    Order Date: 22/01/2025

    Restaurant: Biryani House
    GSTIN: 36AADCB1234M1Z5

    Items:
    Chicken Biryani x1          ₹299.00
    Gulab Jamun x2              ₹99.00

    Subtotal: ₹398.00
    GST: ₹19.90
    Delivery Fee: ₹30.00
    Packaging Charge: ₹15.00
    Total: ₹462.90

    Payment: UPI
    """


@pytest.fixture
def sample_bank_text():
    return """
    HDFC Bank
    Account Statement
    Account Holder: Nani Vishnu
    Account Number: 5020 1234 5678 9012
    IFSC: HDFC0001234
    Savings Account

    Period: 01/01/2025 To 31/01/2025
    Opening Balance: 45,230.50

    01/01/2025 UPI-NANI-TO-SWIGGY          500.00                44,730.50
    05/01/2025 SALARY JAN 2025                      50,000.00    94,730.50
    10/01/2025 AMAZON PAY TOPUP             1,000.00             93,730.50
    15/01/2025 RENT TRANSFER                15,000.00            78,730.50

    Closing Balance: 78,730.50
    """
