"""
Tests for bank statement extraction.
Run: pytest tests/test_bank.py -v
"""


class TestBankExtraction:
    def test_bank_name(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.bank_name == "HDFC Bank"

    def test_account_holder(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.account_holder is not None
        assert "Nani" in data.account_holder

    def test_account_number(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.account_number is not None

    def test_ifsc(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.ifsc_code == "HDFC0001234"

    def test_account_type(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.account_type == "Savings"

    def test_opening_balance(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.opening_balance == 45230.50

    def test_closing_balance(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.closing_balance == 78730.50

    def test_transactions_found(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        assert data.transactions is not None
        assert data.transaction_count > 0

    def test_analytics(self, bank_extractor, sample_bank_text):
        data = bank_extractor.extract(sample_bank_text)
        if data.total_debits:
            assert data.total_debits > 0
        if data.total_credits:
            assert data.total_credits > 0
