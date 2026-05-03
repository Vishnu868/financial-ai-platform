"""
Tests for post-extraction validator — updated for flat schema.
Run: pytest tests/test_validation.py -v
"""

from app.models.schemas import InvoiceData


class TestGSTINValidation:
    def test_invalid_format_too_short(self, validator):
        result = validator._validate_gstin("29AABCT")
        assert result is not None

    def test_invalid_state_code(self, validator):
        result = validator._validate_gstin("99AABCT1234F1Z5")
        assert result is not None
        assert "state" in result.lower()

    def test_valid_format(self, validator):
        result = validator._validate_gstin("29AABCT1234F1Z5")
        # Either passes or fails only on checksum
        if result:
            assert "checksum" in result.lower() or "check digit" in result.lower()


class TestTaxMathValidation:
    def test_matching_tax(self, validator):
        data = InvoiceData(cgst_amount=50.0, sgst_amount=50.0, total_tax=100.0)
        result = validator._validate_tax_math(data)
        assert result is None

    def test_mismatch(self, validator):
        data = InvoiceData(cgst_amount=50.0, sgst_amount=50.0, total_tax=200.0)
        result = validator._validate_tax_math(data)
        assert result is not None
        assert "math" in result.lower() or "error" in result.lower()

    def test_cgst_not_equal_sgst(self, validator):
        data = InvoiceData(cgst_amount=50.0, sgst_amount=80.0, total_tax=130.0)
        result = validator._validate_tax_math(data)
        assert result is not None
        assert "CGST" in result and "SGST" in result

    def test_igst_with_cgst_conflict(self, validator):
        data = InvoiceData(cgst_amount=50.0, igst_amount=100.0, total_tax=150.0)
        result = validator._validate_tax_math(data)
        assert result is not None
        assert "Both IGST" in result

    def test_igst_only_valid(self, validator):
        data = InvoiceData(igst_amount=100.0, total_tax=100.0)
        result = validator._validate_tax_math(data)
        assert result is None

    def test_within_tolerance(self, validator):
        data = InvoiceData(cgst_amount=50.0, sgst_amount=50.0, total_tax=100.50)
        result = validator._validate_tax_math(data)
        assert result is None


class TestTotalCrossCheck:
    def test_correct_total(self, validator):
        data = InvoiceData(
            subtotal=1000.0, total_tax=180.0,
            discount=100.0, delivery_charge=50.0,
            total_amount=1130.0,
        )
        result = validator._validate_total(data)
        assert result is None

    def test_wrong_total(self, validator):
        data = InvoiceData(
            subtotal=1000.0, total_tax=180.0,
            total_amount=5000.0,
        )
        result = validator._validate_total(data)
        assert result is not None
        assert "cross-check" in result.lower()

    def test_no_subtotal_skips(self, validator):
        data = InvoiceData(total_amount=1000.0)
        result = validator._validate_total(data)
        assert result is None

    def test_with_packaging(self, validator):
        data = InvoiceData(
            subtotal=398.0, total_tax=19.90,
            delivery_charge=30.0, packaging_charge=15.0,
            total_amount=462.90,
        )
        result = validator._validate_total(data)
        assert result is None


class TestDateValidation:
    def test_valid_dd_mm_yyyy(self, validator):
        assert validator._validate_date("15/01/2025") is None

    def test_valid_yyyy_mm_dd(self, validator):
        assert validator._validate_date("2025-01-15") is None

    def test_future_date(self, validator):
        result = validator._validate_date("01/01/2030")
        assert result is not None
        assert "future" in result


class TestFullValidation:
    def test_empty_invoice_warns(self, validator):
        data = InvoiceData()
        passed, warnings = validator.validate(data)
        assert not passed
        assert len(warnings) > 0

    def test_negative_total_warns(self, validator):
        data = InvoiceData(total_amount=-500.0, fields_extracted=5)
        passed, warnings = validator.validate(data)
        assert not passed
        assert any("negative" in w.lower() for w in warnings)

    def test_huge_total_warns(self, validator):
        data = InvoiceData(total_amount=50_000_000.0, fields_extracted=5)
        passed, warnings = validator.validate(data)
        assert any("crore" in w.lower() for w in warnings)
