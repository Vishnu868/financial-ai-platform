"""
Tests for invoice extraction — updated for flat 22-field schema.
Run: pytest tests/test_extraction.py -v
"""

from app.models.schemas import DocumentPlatform, InvoiceData


class TestPlatformDetection:
    def test_amazon(self, extractor, sample_amazon_text):
        assert extractor.detect_platform(sample_amazon_text) == DocumentPlatform.AMAZON

    def test_flipkart(self, extractor, sample_flipkart_text):
        assert extractor.detect_platform(sample_flipkart_text) == DocumentPlatform.FLIPKART

    def test_swiggy(self, extractor, sample_swiggy_text):
        assert extractor.detect_platform(sample_swiggy_text) == DocumentPlatform.SWIGGY

    def test_unknown(self, extractor):
        assert extractor.detect_platform("random text here") == DocumentPlatform.UNKNOWN


class TestAmazonExtraction:
    def test_order_number(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.order_number == "408-1234567-8901234"

    def test_invoice_number(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.invoice_number is not None
        assert "12345678" in data.invoice_number

    def test_total_amount(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.total_amount == 1294.46

    def test_subtotal(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.subtotal == 1097.0

    def test_seller_gst(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.seller_gst == "29AABCT1234F1Z5"

    def test_seller_name(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.seller_name is not None
        assert "TechStore" in data.seller_name

    def test_cgst(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.cgst_amount == 98.73

    def test_sgst(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.sgst_amount == 98.73

    def test_total_tax(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.total_tax is not None
        assert abs(data.total_tax - 197.46) < 0.1

    def test_payment_method(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.payment_method == "Amazon Pay"

    def test_invoice_date(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.invoice_date is not None

    def test_buyer_phone(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.buyer_phone is not None

    def test_place_of_supply(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.place_of_supply is not None
        assert "Telangana" in data.place_of_supply

    def test_invoice_type(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.invoice_type == "Tax Invoice"

    def test_seller_pan(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.seller_pan is not None

    def test_fields_extracted(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        assert data.fields_extracted >= 6


class TestFlipkartExtraction:
    def test_order_number(self, extractor, sample_flipkart_text):
        data = extractor.extract_all_fields(sample_flipkart_text, DocumentPlatform.FLIPKART)
        assert data.order_number == "OD123456789012"

    def test_total(self, extractor, sample_flipkart_text):
        data = extractor.extract_all_fields(sample_flipkart_text, DocumentPlatform.FLIPKART)
        assert data.total_amount == 1768.82

    def test_discount(self, extractor, sample_flipkart_text):
        data = extractor.extract_all_fields(sample_flipkart_text, DocumentPlatform.FLIPKART)
        assert data.discount == 500.0


class TestSwiggyExtraction:
    def test_delivery_charge(self, extractor, sample_swiggy_text):
        data = extractor.extract_all_fields(sample_swiggy_text, DocumentPlatform.SWIGGY)
        assert data.delivery_charge == 30.0

    def test_packaging(self, extractor, sample_swiggy_text):
        data = extractor.extract_all_fields(sample_swiggy_text, DocumentPlatform.SWIGGY)
        assert data.packaging_charge == 15.0

    def test_total(self, extractor, sample_swiggy_text):
        data = extractor.extract_all_fields(sample_swiggy_text, DocumentPlatform.SWIGGY)
        assert data.total_amount == 462.90


class TestConfidence:
    def test_high_confidence(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        conf = extractor.calculate_confidence(data, 0.9)
        assert conf > 0.5

    def test_low_confidence_empty(self, extractor):
        data = InvoiceData()
        conf = extractor.calculate_confidence(data, 0.3)
        assert conf < 0.2

    def test_confidence_range(self, extractor, sample_amazon_text):
        data = extractor.extract_all_fields(sample_amazon_text, DocumentPlatform.AMAZON)
        conf = extractor.calculate_confidence(data, 0.85)
        assert 0.0 <= conf <= 1.0
