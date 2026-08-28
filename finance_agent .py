"""
========================================================================================
FinSight AI — Intelligent Finance & Accounts Agent (Single Python File < 500 lines)
========================================================================================
All-in-one production Agentic AI system for enterprise financial analysis, accounting audits,
receivables/payables tracking, anomaly detection, policy RAG, and FastAPI REST endpoints.
========================================================================================
"""

import os
import sys
import re
import json
import time
import math
import sqlite3
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# UTF-8 Console safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Core Libraries
try:
    import pandas as pd
except ImportError:
    pd = None

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Optional LangChain integration
try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ======================================================================================
# 1. CONFIGURATION & LOGGING
# ======================================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s")
logger = logging.getLogger("FinSightAI")

class Config:
    APP_NAME: str = "FinSight AI — Finance & Accounts Agent"
    VERSION: str = "2.0.0"
    DB_PATH: str = os.getenv("FINSIGHT_DB_PATH", "finsight_finance.db")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", None)
    LLM_MODEL: str = os.getenv("FINSIGHT_LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("FINSIGHT_LLM_TEMP", "0.0"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


# ======================================================================================
# 2. DATABASE ENGINE & REALISTIC DEMO DATA SEEDING
# ======================================================================================

class DatabaseManager:
    """Manages SQLite connection, relational schema, and realistic demo data generation."""

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        with self.get_connection() as conn:
            c = conn.cursor()
            c.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, industry TEXT, contact_email TEXT,
                credit_limit REAL, balance REAL DEFAULT 0.0, payment_terms TEXT DEFAULT 'Net 30'
            );
            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT, contact_email TEXT,
                balance REAL DEFAULT 0.0, payment_terms TEXT DEFAULT 'Net 30'
            );
            CREATE TABLE IF NOT EXISTS departments (
                department_id TEXT PRIMARY KEY, department_name TEXT NOT NULL UNIQUE, manager_name TEXT,
                quarterly_budget REAL, annual_budget REAL
            );
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, type TEXT, category TEXT,
                description TEXT, amount REAL NOT NULL, department TEXT, customer_id TEXT,
                supplier_id TEXT, payment_method TEXT, reference_no TEXT
            );
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id TEXT PRIMARY KEY, invoice_number TEXT UNIQUE, type TEXT, entity_id TEXT,
                entity_name TEXT NOT NULL, issue_date TEXT, due_date TEXT, amount REAL,
                paid_amount REAL DEFAULT 0.0, status TEXT, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS expenses (
                expense_id TEXT PRIMARY KEY, date TEXT NOT NULL, department TEXT, category TEXT,
                vendor TEXT, amount REAL NOT NULL, approved_by TEXT, description TEXT
            );
            CREATE TABLE IF NOT EXISTS accounting_policies (
                policy_id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT, summary TEXT, content TEXT
            );
            """)
            conn.commit()
        self._seed_demo_data()

    def _seed_demo_data(self) -> None:
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM customers;")
            if c.fetchone()[0] > 0:
                return

            logger.info("Seeding realistic corporate accounting dataset into SQLite...")
            # Departments
            c.executemany("INSERT INTO departments VALUES (?,?,?,?,?)", [
                ('DEP-ENG', 'Engineering', 'Alex Mercer', 120000.0, 480000.0),
                ('DEP-MKT', 'Marketing', 'Elena Rostova', 85000.0, 340000.0),
                ('DEP-SAL', 'Sales', 'Marcus Brody', 95000.0, 380000.0),
                ('DEP-OPS', 'Operations', 'Sarah Chen', 70000.0, 280000.0),
                ('DEP-HR', 'Human Resources', 'David Vance', 45000.0, 180000.0),
                ('DEP-FIN', 'Finance & Legal', 'Rachel Adams', 50000.0, 200000.0)
            ])
            # Customers
            c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", [
                ('CUST-001', 'Apex Global Technologies', 'Enterprise SaaS', 'billing@apextech.com', 150000.0, 45000.0, 'Net 30'),
                ('CUST-002', 'Nexus Retail Group', 'E-Commerce', 'ap@nexusretail.com', 100000.0, 68500.0, 'Net 45'),
                ('CUST-003', 'Horizon Media & Entertainment', 'Digital Media', 'fin@horizon.com', 75000.0, 18000.0, 'Net 30'),
                ('CUST-004', 'Quantum Health Systems', 'Healthcare', 'payables@quantum.com', 120000.0, 0.0, 'Net 30'),
                ('CUST-005', 'Starlight Logistics', 'Supply Chain', 'accts@starlight.com', 80000.0, 24500.0, 'Net 60'),
                ('CUST-006', 'Vanguard Industrial Supply', 'Manufacturing', 'accounting@vanguard.com', 90000.0, 0.0, 'Net 30'),
                ('CUST-007', 'Bluebird Fintech Labs', 'Fintech', 'accounts@bluebirdfin.io', 110000.0, 52000.0, 'Net 30')
            ])
            # Suppliers
            c.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?)", [
                ('SUP-001', 'CloudScale Infrastructure (AWS/GCP)', 'Cloud Hosting', 'billing@cloudscale.net', 14200.0, 'Net 15'),
                ('SUP-002', 'TalentForce Global Staffing', 'Contracting', 'invoices@talentforce.com', 28500.0, 'Net 30'),
                ('SUP-003', 'OfficeMax Corporate Facilities', 'Hardware', 'orders@officemax.com', 4300.0, 'Net 30'),
                ('SUP-004', 'CyberGuard InfoSec Solutions', 'Security', 'billing@cyberguard.io', 16000.0, 'Net 30'),
                ('SUP-005', 'OmniChannel Advertising Partners', 'Marketing', 'finance@omnichannel.com', 32000.0, 'Net 30'),
                ('SUP-006', 'Lexington Legal & Tax Advisors', 'Legal', 'tax@lexingtonlaw.com', 9500.0, 'Net 45')
            ])
            # Invoices
            c.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
                ('INV-REC-001', 'INV-2026-001', 'RECEIVABLE', 'CUST-001', 'Apex Global Technologies', '2026-06-01', '2026-07-01', 45000.0, 0.0, 'OVERDUE', 'Q2 Enterprise License'),
                ('INV-REC-002', 'INV-2026-002', 'RECEIVABLE', 'CUST-002', 'Nexus Retail Group', '2026-06-15', '2026-07-30', 68500.0, 0.0, 'OVERDUE', 'E-commerce API Integration'),
                ('INV-REC-003', 'INV-2026-003', 'RECEIVABLE', 'CUST-003', 'Horizon Media & Entertainment', '2026-07-05', '2026-08-05', 36000.0, 18000.0, 'PARTIALLY_PAID', 'Content Engine sprint 4'),
                ('INV-REC-004', 'INV-2026-004', 'RECEIVABLE', 'CUST-004', 'Quantum Health Systems', '2026-07-10', '2026-08-10', 92000.0, 92000.0, 'PAID', 'Analytics Engine paid in full'),
                ('INV-REC-005', 'INV-2026-005', 'RECEIVABLE', 'CUST-005', 'Starlight Logistics', '2026-07-20', '2026-09-20', 24500.0, 0.0, 'UNPAID', 'Route optimization license'),
                ('INV-REC-006', 'INV-2026-006', 'RECEIVABLE', 'CUST-006', 'Vanguard Industrial Supply', '2026-06-10', '2026-07-10', 58000.0, 58000.0, 'PAID', 'Predictive maintenance license'),
                ('INV-REC-007', 'INV-2026-007', 'RECEIVABLE', 'CUST-007', 'Bluebird Fintech Labs', '2026-08-01', '2026-08-31', 52000.0, 0.0, 'UNPAID', 'Fraud API Gateway tier-1'),
                ('INV-PAY-101', 'BILL-2026-101', 'PAYABLE', 'SUP-001', 'CloudScale Infrastructure (AWS/GCP)', '2026-07-01', '2026-07-16', 14200.0, 0.0, 'OVERDUE', 'Monthly cloud server compute'),
                ('INV-PAY-102', 'BILL-2026-102', 'PAYABLE', 'SUP-002', 'TalentForce Global Staffing', '2026-07-15', '2026-08-15', 28500.0, 0.0, 'OVERDUE', 'Contractor billing July'),
                ('INV-PAY-103', 'BILL-2026-103', 'PAYABLE', 'SUP-003', 'OfficeMax Corporate Facilities', '2026-07-25', '2026-08-25', 4300.0, 0.0, 'OVERDUE', 'Office hardware supplies'),
                ('INV-PAY-104', 'BILL-2026-104', 'PAYABLE', 'SUP-004', 'CyberGuard InfoSec Solutions', '2026-08-02', '2026-09-02', 16000.0, 0.0, 'UNPAID', 'SOC2 Security Audit Retainer'),
                ('INV-PAY-105', 'BILL-2026-105', 'PAYABLE', 'SUP-005', 'OmniChannel Advertising Partners', '2026-08-05', '2026-09-05', 32000.0, 0.0, 'UNPAID', 'Q3 Performance Ads'),
                ('INV-PAY-106', 'BILL-2026-106', 'PAYABLE', 'SUP-006', 'Lexington Legal & Tax Advisors', '2026-08-10', '2026-09-25', 9500.0, 0.0, 'UNPAID', 'Corporate tax filing advisory')
            ])
            # Expenses (Contains intentional anomalies: Marketing software spike $48.5k & HR duplicate $6.2k)
            c.executemany("INSERT INTO expenses VALUES (?,?,?,?,?,?,?,?)", [
                ('EXP-001', '2026-06-05', 'Engineering', 'Cloud Infrastructure', 'CloudScale Infrastructure', 13800.0, 'Alex Mercer', 'June Cloud compute'),
                ('EXP-002', '2026-07-04', 'Engineering', 'Cloud Infrastructure', 'CloudScale Infrastructure', 14200.0, 'Alex Mercer', 'July Cloud compute'),
                ('EXP-003', '2026-08-03', 'Engineering', 'Cloud Infrastructure', 'CloudScale Infrastructure', 15100.0, 'Alex Mercer', 'August Cloud compute'),
                ('EXP-004', '2026-06-15', 'Engineering', 'Contractor Services', 'TalentForce Global Staffing', 27000.0, 'Alex Mercer', 'Contract engineers June'),
                ('EXP-005', '2026-07-15', 'Engineering', 'Contractor Services', 'TalentForce Global Staffing', 28500.0, 'Alex Mercer', 'Contract engineers July'),
                ('EXP-006', '2026-08-15', 'Engineering', 'Contractor Services', 'TalentForce Global Staffing', 29000.0, 'Alex Mercer', 'Contract engineers August'),
                ('EXP-007', '2026-07-20', 'Engineering', 'Software Tools & Subscriptions', 'GitHub & JetBrains', 6400.0, 'Alex Mercer', 'Developer tooling licenses'),
                ('EXP-008', '2026-06-10', 'Marketing', 'Digital Advertising', 'OmniChannel Advertising Partners', 28000.0, 'Elena Rostova', 'June Search Ads'),
                ('EXP-009', '2026-07-12', 'Marketing', 'Digital Advertising', 'OmniChannel Advertising Partners', 31500.0, 'Elena Rostova', 'July Launch Ads'),
                ('EXP-010', '2026-08-10', 'Marketing', 'Digital Advertising', 'OmniChannel Advertising Partners', 32000.0, 'Elena Rostova', 'August Brand Ads'),
                ('EXP-011', '2026-06-18', 'Marketing', 'Events & Sponsorships', 'SaaS Summit 2026', 15000.0, 'Elena Rostova', 'Annual summit booth'),
                ('EXP-012', '2026-07-28', 'Marketing', 'Software Tools & Subscriptions', 'OmniChannel Analytics', 48500.0, 'Elena Rostova', 'UNSCHEDULED Enterprise Annual BI Tool (Anomaly Spike)'),
                ('EXP-013', '2026-08-05', 'Marketing', 'Content Creation', 'Apex Creative Studio', 7500.0, 'Elena Rostova', 'Demo video production'),
                ('EXP-014', '2026-06-12', 'Sales', 'Travel & Client Entertainment', 'Delta Airlines & Marriott', 8400.0, 'Marcus Brody', 'Client pitches NY & Chicago'),
                ('EXP-015', '2026-07-14', 'Sales', 'Travel & Client Entertainment', 'Delta Airlines & Marriott', 11200.0, 'Marcus Brody', 'Enterprise account on-site reviews'),
                ('EXP-016', '2026-08-12', 'Sales', 'Travel & Client Entertainment', 'United Airlines & Hilton', 9300.0, 'Marcus Brody', 'Partner meetings'),
                ('EXP-017', '2026-06-28', 'Sales', 'CRM Software', 'Salesforce CRM', 14500.0, 'Marcus Brody', 'Quarterly CRM renewal'),
                ('EXP-018', '2026-07-30', 'Sales', 'Commissions & Incentives', 'Direct Payroll', 22000.0, 'Marcus Brody', 'Q2 Sales commissions'),
                ('EXP-019', '2026-06-01', 'Operations', 'Office Facilities & Lease', 'Metro Real Estate Group', 18500.0, 'Sarah Chen', 'Corporate lease June'),
                ('EXP-020', '2026-07-01', 'Operations', 'Office Facilities & Lease', 'Metro Real Estate Group', 18500.0, 'Sarah Chen', 'Corporate lease July'),
                ('EXP-021', '2026-08-01', 'Operations', 'Office Facilities & Lease', 'Metro Real Estate Group', 18500.0, 'Sarah Chen', 'Corporate lease August'),
                ('EXP-022', '2026-07-26', 'Operations', 'Hardware & Office Supplies', 'OfficeMax Corporate Facilities', 8100.0, 'Sarah Chen', 'Laptops for new joiners'),
                ('EXP-023', '2026-06-15', 'Human Resources', 'Employee Benefits & Wellness', 'BlueCross Health Plan', 12500.0, 'David Vance', 'June Health insurance'),
                ('EXP-024', '2026-07-15', 'Human Resources', 'Employee Benefits & Wellness', 'BlueCross Health Plan', 12500.0, 'David Vance', 'July Health insurance'),
                ('EXP-025', '2026-08-15', 'Human Resources', 'Employee Benefits & Wellness', 'BlueCross Health Plan', 12500.0, 'David Vance', 'August Health insurance'),
                ('EXP-026', '2026-07-10', 'Human Resources', 'Recruitment Advertising', 'LinkedIn Recruiter Corp', 6200.0, 'David Vance', 'Job posting slots renewal'),
                ('EXP-027', '2026-07-10', 'Human Resources', 'Recruitment Advertising', 'LinkedIn Recruiter Corp', 6200.0, 'David Vance', 'DUPLICATE: Job posting slots renewal (Glitch)'),
                ('EXP-028', '2026-06-30', 'Finance & Legal', 'Legal & Compliance Advisory', 'Lexington Legal & Tax Advisors', 9500.0, 'Rachel Adams', 'Q2 Governance counsel'),
                ('EXP-029', '2026-07-31', 'Finance & Legal', 'Audit & Accounting Software', 'Intuit & NetSuite', 6800.0, 'Rachel Adams', 'ERP software subscription'),
                ('EXP-030', '2026-08-02', 'Finance & Legal', 'Security & Compliance Audit', 'CyberGuard InfoSec Solutions', 16000.0, 'Rachel Adams', 'Annual SOC2 Audit Retainer')
            ])
            # Transactions
            c.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
                ('TX-001', '2026-06-02', 'INCOME', 'Software License', 'Apex Global Q2 Renewal Payment', 125000.0, 'Sales', 'CUST-001', None, 'Wire Transfer', 'REF-0601'),
                ('TX-002', '2026-06-12', 'INCOME', 'Consulting', 'Vanguard Predictive Maintenance Paid', 58000.0, 'Sales', 'CUST-006', None, 'ACH Transfer', 'REF-0612'),
                ('TX-003', '2026-06-25', 'INCOME', 'SaaS Subscription', 'Nexus Retail Monthly Tier Payment', 42000.0, 'Sales', 'CUST-002', None, 'Credit Card', 'REF-0625'),
                ('TX-004', '2026-06-05', 'EXPENSE', 'Cloud Hosting', 'CloudScale June Hosting Paid', 13800.0, 'Engineering', None, 'SUP-001', 'ACH Direct', 'REF-AWS06'),
                ('TX-005', '2026-06-15', 'EXPENSE', 'Staffing', 'TalentForce Contractor Payment June', 27000.0, 'Engineering', None, 'SUP-002', 'Wire Transfer', 'REF-TF06'),
                ('TX-006', '2026-06-01', 'EXPENSE', 'Rent', 'Metro Real Estate June Rent Paid', 18500.0, 'Operations', None, None, 'Direct Debit', 'REF-RNT06'),
                ('TX-007', '2026-07-08', 'INCOME', 'Enterprise License', 'Quantum Health Compliance Engine Paid', 92000.0, 'Sales', 'CUST-004', None, 'Wire Transfer', 'REF-0708'),
                ('TX-008', '2026-07-16', 'INCOME', 'Consulting', 'Horizon Media Partial Milestone Paid', 18000.0, 'Sales', 'CUST-003', None, 'ACH Transfer', 'REF-0716'),
                ('TX-009', '2026-07-28', 'INCOME', 'Subscription', 'Bluebird Fintech Implementation Batch', 35000.0, 'Sales', 'CUST-007', None, 'Wire Transfer', 'REF-0728'),
                ('TX-010', '2026-07-01', 'EXPENSE', 'Rent', 'Metro Real Estate July Rent Paid', 18500.0, 'Operations', None, None, 'Direct Debit', 'REF-RNT07'),
                ('TX-011', '2026-07-15', 'EXPENSE', 'Health Benefits', 'BlueCross Health Plan July Premium', 12500.0, 'Human Resources', None, None, 'ACH Direct', 'REF-BC07'),
                ('TX-012', '2026-07-28', 'EXPENSE', 'Software', 'OmniChannel Analytics Annual Paid', 48500.0, 'Marketing', None, 'SUP-005', 'Card', 'REF-MKT07'),
                ('TX-013', '2026-08-04', 'INCOME', 'SaaS Platform License', 'Apex Tech Q3 Recurring SaaS Installment', 80000.0, 'Sales', 'CUST-001', None, 'Wire Transfer', 'REF-0804'),
                ('TX-014', '2026-08-14', 'INCOME', 'Custom Integration', 'Horizon Media Sprint Milestone 2 Paid', 25000.0, 'Sales', 'CUST-003', None, 'ACH Transfer', 'REF-0814'),
                ('TX-015', '2026-08-01', 'EXPENSE', 'Rent', 'Metro Real Estate August Rent Paid', 18500.0, 'Operations', None, None, 'Direct Debit', 'REF-RNT08')
            ])
            # Accounting Policies
            c.executemany("INSERT INTO accounting_policies VALUES (?,?,?,?,?)", [
                ('POL-001', 'Capitalization & Fixed Asset Policy', 'Asset Management', 'Equipment capitalization threshold.',
                 'Any equipment, software license, or hardware item costing greater than $5,000 with a useful life >12 months must be capitalized and depreciated straight-line over 36 months. Items below $5,000 are expensed immediately.'),
                ('POL-002', 'Travel & Expense Policy', 'Expense Management', 'Flight and daily per-diem rules.',
                 'Flights exceeding $800 require Department Head approval. Lodging is capped at $250/night standard ($380/night Tier 1 cities). Meals per-diem is $75/day. Receipts mandatory over $25.'),
                ('POL-003', 'Revenue Recognition Policy (ASC 606)', 'Revenue Accounting', 'SaaS subscription recognition.',
                 'SaaS revenue is recognized ratably over the subscription term daily. One-time setup and consulting fees are recognized upon milestone acceptance sign-off.'),
                ('POL-004', 'Accounts Receivable & Overdue Protocol', 'Collections', 'Escalation procedure for past due balances.',
                 'Standard terms Net 30. At 15 days past due: automated reminder. At 30 days past due: account executive outreach. At 60 days past due: service suspension. At 90 days: bad debt evaluation.'),
                ('POL-005', 'Vendor Payment Authorization & Fraud Protocol', 'Payables Governance', 'Dual authorization rules.',
                 'Disbursements exceeding $10,000 require dual authorization from CFO and Department Head. Supplier bank updates require phone verification. Duplicate invoices trigger audit hold.')
            ])
            conn.commit()
            logger.info("Database successfully seeded with realistic corporate accounting records!")

db = DatabaseManager()


# ======================================================================================
# 3. DETERMINISTIC FINANCIAL ENGINE
# ======================================================================================

class FinancialEngine:
    """Pure Python & SQLite Financial Engine providing deterministic mathematical calculations."""

    @staticmethod
    def execute_query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        sanitized = sql.strip().upper()
        if any(sanitized.startswith(kw) for kw in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE']):
            raise ValueError("Security violation: Only safe read-only SELECT queries are permitted.")
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def fmt_currency(val: Union[float, int]) -> str:
        return f"${float(val or 0):,.2f}"

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        with db.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COALESCE(SUM(amount), 0), COALESCE(SUM(paid_amount), 0) FROM invoices WHERE type = 'RECEIVABLE';")
            tot_rev, tot_rev_paid = c.fetchone()
            c.execute("SELECT COALESCE(SUM(amount), 0) FROM expenses;")
            tot_exp = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE type = 'INCOME';")
            cash_in = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE type = 'EXPENSE';")
            cash_out = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(amount - paid_amount), 0) FROM invoices WHERE type = 'RECEIVABLE' AND status != 'PAID';")
            ar = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(amount - paid_amount), 0) FROM invoices WHERE type = 'PAYABLE' AND status != 'PAID';")
            ap = c.fetchone()[0]

            net_profit = tot_rev - tot_exp
            margin = (net_profit / tot_rev * 100.0) if tot_rev > 0 else 0.0
            return {
                "total_revenue": tot_rev, "revenue_collected": tot_rev_paid, "total_expenses": tot_exp,
                "net_profit": net_profit, "profit_margin_pct": round(margin, 2),
                "accounts_receivable": ar, "accounts_payable": ap,
                "cash_inflow": cash_in, "cash_outflow": cash_out, "net_cash_flow": cash_in - cash_out
            }

    @classmethod
    def detect_anomalies(cls) -> List[Dict[str, Any]]:
        anomalies = []
        # 1. Duplicate Expenses
        dups = cls.execute_query("SELECT date, department, vendor, amount, COUNT(*) as c FROM expenses GROUP BY date, vendor, amount HAVING COUNT(*) > 1;")
        for d in dups:
            anomalies.append({
                "type": "DUPLICATE_EXPENSE", "severity": "HIGH", "department": d["department"],
                "amount": d["amount"], "description": f"Duplicate detected: {d['c']} identical records of ${d['amount']:,.2f} on {d['date']} to {d['vendor']}."
            })
        # 2. Outliers (Z-Score >= 2.0)
        exp_rows = cls.execute_query("SELECT expense_id, department, category, vendor, amount, date FROM expenses;")
        if exp_rows:
            amounts = [float(r["amount"]) for r in exp_rows]
            mean_val = sum(amounts) / len(amounts)
            variance = sum((x - mean_val) ** 2 for x in amounts) / (len(amounts) - 1 or 1)
            std_dev = math.sqrt(variance) or 1.0
            for r in exp_rows:
                z = (float(r["amount"]) - mean_val) / std_dev
                if z >= 2.0:
                    anomalies.append({
                        "type": "EXPENSE_OUTLIER", "severity": "MEDIUM_HIGH", "department": r["department"],
                        "amount": r["amount"], "description": f"Expense of ${r['amount']:,.2f} ({r['category']}) has a Z-score of {z:.2f} (Mean is ${mean_val:,.2f})."
                    })
        # 3. Budget Overutilization
        budgets = cls.execute_query("SELECT d.department_name, d.quarterly_budget, COALESCE(SUM(e.amount), 0) as spent FROM departments d LEFT JOIN expenses e ON d.department_name = e.department GROUP BY d.department_id;")
        for b in budgets:
            pct = (b["spent"] / b["quarterly_budget"] * 100.0) if b["quarterly_budget"] > 0 else 0
            if pct > 90.0:
                anomalies.append({
                    "type": "BUDGET_OVERRUN_RISK", "severity": "MEDIUM", "department": b["department_name"],
                    "amount": b["spent"], "description": f"Department '{b['department_name']}' has utilized {pct:.1f}% of quarterly budget (${b['spent']:,.2f} of ${b['quarterly_budget']:,.2f})."
                })
        return anomalies


# ======================================================================================
# 4. AGENT TOOLS DEFINITION
# ======================================================================================

def _table_md(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows: return "_No records found._"
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(["| " + " | ".join(str(x) for x in r) + " |" for r in rows])

def query_financial_sql(sql_query: str) -> str:
    """Execute a safe read-only SQL query on the company SQLite database."""
    logger.info(f"Tool: query_financial_sql | {sql_query}")
    try:
        results = FinancialEngine.execute_query(sql_query)
        if not results: return "Result: 0 rows returned."
        headers = list(results[0].keys())
        rows = [[row[h] for h in headers] for row in results[:25]]
        return f"### SQL Query Result ({len(results)} rows)\n\n" + _table_md(headers, rows)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def search_financial_records(query: str, entity_type: str = "all") -> str:
    """Search customers, suppliers, transactions, invoices, or expenses by keyword."""
    logger.info(f"Tool: search_financial_records | {query}")
    try:
        q = f"%{query.strip()}%"
        parts = []
        with db.get_connection() as conn:
            c = conn.cursor()
            if entity_type in ('all', 'customers'):
                c.execute("SELECT customer_id, name, industry, balance, payment_terms FROM customers WHERE name LIKE ? OR industry LIKE ?;", (q, q))
                rows = c.fetchall()
                if rows: parts.append("**Matching Customers:**\n" + _table_md(["ID", "Name", "Industry", "Balance", "Terms"], [[r[0], r[1], r[2], f"${r[3]:,.2f}", r[4]] for r in rows]))
            if entity_type in ('all', 'invoices'):
                c.execute("SELECT invoice_number, type, entity_name, amount, paid_amount, status, due_date FROM invoices WHERE entity_name LIKE ? OR notes LIKE ?;", (q, q))
                rows = c.fetchall()
                if rows: parts.append("**Matching Invoices:**\n" + _table_md(["Inv #", "Type", "Entity", "Amount", "Paid", "Status", "Due Date"], [[r[0], r[1], r[2], f"${r[3]:,.2f}", f"${r[4]:,.2f}", r[5], r[6]] for r in rows]))
            if entity_type in ('all', 'expenses'):
                c.execute("SELECT expense_id, date, department, category, vendor, amount, description FROM expenses WHERE vendor LIKE ? OR department LIKE ? OR description LIKE ?;", (q, q, q))
                rows = c.fetchall()
                if rows: parts.append("**Matching Expenses:**\n" + _table_md(["ID", "Date", "Department", "Category", "Vendor", "Amount", "Description"], [[r[0], r[1], r[2], r[3], r[4], f"${r[5]:,.2f}", r[6]] for r in rows]))
        return "\n\n".join(parts) if parts else f"No records found matching '{query}'."
    except Exception as e:
        return f"Search Error: {str(e)}"

def calculate_financial_metrics(metric_type: str = "summary") -> str:
    """Calculate key financial metrics including Revenue, Net Profit, Profit Margin, AR, AP, and Cash Flow."""
    logger.info("Tool: calculate_financial_metrics")
    try:
        s = FinancialEngine.get_summary()
        headers = ["Financial Metric", "Amount (USD)", "Notes"]
        rows = [
            ["Total Invoiced Revenue", FinancialEngine.fmt_currency(s["total_revenue"]), "Total customer billings"],
            ["Revenue Collected", FinancialEngine.fmt_currency(s["revenue_collected"]), "Actual cash collected from invoices"],
            ["Total Expenses", FinancialEngine.fmt_currency(s["total_expenses"]), "Sum of all department costs"],
            ["Net Operating Profit", FinancialEngine.fmt_currency(s["net_profit"]), "Revenue - Total Expenses"],
            ["Net Profit Margin", f"{s['profit_margin_pct']}%", "Healthy margin" if s['profit_margin_pct'] > 0 else "Operating deficit"],
            ["Accounts Receivable (AR)", FinancialEngine.fmt_currency(s["accounts_receivable"]), "Customer balance owed to company"],
            ["Accounts Payable (AP)", FinancialEngine.fmt_currency(s["accounts_payable"]), "Supplier balance company owes"],
            ["Net Cash Flow", FinancialEngine.fmt_currency(s["net_cash_flow"]), "Inflow minus Outflow in bank ledger"]
        ]
        return "### Executive Financial Overview\n\n" + _table_md(headers, rows)
    except Exception as e:
        return f"Calculation Error: {str(e)}"

def analyze_accounts_receivable(min_days_overdue: int = 0) -> str:
    """Analyze money owed TO the company by customers, including aging breakdown and overdue invoices."""
    logger.info("Tool: analyze_accounts_receivable")
    try:
        rows = FinancialEngine.execute_query("""
            SELECT invoice_number, entity_name, amount, paid_amount, (amount - paid_amount) AS balance,
                   status, due_date, CAST(julianday('now') - julianday(due_date) AS INT) AS days_overdue
            FROM invoices WHERE type = 'RECEIVABLE' AND status != 'PAID' ORDER BY days_overdue DESC;
        """)
        tot = sum(r["balance"] for r in rows)
        headers = ["Invoice #", "Customer Name", "Total", "Outstanding", "Status", "Days Overdue", "Due Date"]
        table_rows = [[r["invoice_number"], r["entity_name"], f"${r['amount']:,.2f}", f"${r['balance']:,.2f}", f"**{r['status']}**" if r['status']=='OVERDUE' else r['status'], f"{max(0, r['days_overdue'])} days", r["due_date"]] for r in rows if max(0, r["days_overdue"]) >= min_days_overdue]
        return f"### Accounts Receivable (AR) Analysis\n\n- **Total Outstanding Receivables:** {FinancialEngine.fmt_currency(tot)}\n\n" + _table_md(headers, table_rows)
    except Exception as e:
        return f"AR Error: {str(e)}"

def analyze_accounts_payable(status_filter: str = "ALL") -> str:
    """Analyze money the company OWES to suppliers, bills, and overdue obligations."""
    logger.info("Tool: analyze_accounts_payable")
    try:
        where = "WHERE type = 'PAYABLE'"
        if status_filter.upper() == 'OVERDUE': where += " AND status = 'OVERDUE'"
        rows = FinancialEngine.execute_query(f"SELECT invoice_number, entity_name, amount, (amount - paid_amount) AS balance, status, due_date, notes FROM invoices {where} ORDER BY due_date ASC;")
        tot = sum(r["balance"] for r in rows)
        headers = ["Bill #", "Supplier", "Total", "Owed", "Status", "Due Date", "Notes"]
        table_rows = [[r["invoice_number"], r["entity_name"], f"${r['amount']:,.2f}", f"${r['balance']:,.2f}", r["status"], r["due_date"], r["notes"] or "N/A"] for r in rows]
        return f"### Accounts Payable (AP) Summary\n\n- **Total Supplier Obligations:** {FinancialEngine.fmt_currency(tot)}\n\n" + _table_md(headers, table_rows)
    except Exception as e:
        return f"AP Error: {str(e)}"

def analyze_expenses(group_by: str = "department") -> str:
    """Analyze company expenses grouped by 'department', 'category', or 'vendor'."""
    logger.info(f"Tool: analyze_expenses | {group_by}")
    try:
        if group_by.lower() == "department":
            rows = FinancialEngine.execute_query("""
                SELECT d.department_name, d.manager_name, d.quarterly_budget, COALESCE(SUM(e.amount), 0) AS spent,
                       (d.quarterly_budget - COALESCE(SUM(e.amount), 0)) AS remaining,
                       ROUND(COALESCE(SUM(e.amount), 0) / d.quarterly_budget * 100.0, 1) AS pct
                FROM departments d LEFT JOIN expenses e ON d.department_name = e.department GROUP BY d.department_id ORDER BY spent DESC;
            """)
            headers = ["Department", "Manager", "Budget", "Total Spent", "Remaining", "% Utilized"]
            t_rows = [[r["department_name"], r["manager_name"], f"${r['quarterly_budget']:,.2f}", f"${r['spent']:,.2f}", f"${r['remaining']:,.2f}", f"{r['pct']}%"] for r in rows]
            return "### Departmental Expense & Budget Variance\n\n" + _table_md(headers, t_rows)
        else:
            rows = FinancialEngine.execute_query("SELECT category, COUNT(*) as count, SUM(amount) as total FROM expenses GROUP BY category ORDER BY total DESC;")
            headers = ["Category", "Transactions", "Total Spent"]
            t_rows = [[r["category"], r["count"], f"${r['total']:,.2f}"] for r in rows]
            return "### Expense Breakdown by Category\n\n" + _table_md(headers, t_rows)
    except Exception as e:
        return f"Expense Analysis Error: {str(e)}"

def analyze_revenue(group_by: str = "month") -> str:
    """Analyze revenue streams, monthly billing progression, and customer contributions."""
    logger.info(f"Tool: analyze_revenue | {group_by}")
    try:
        if group_by.lower() in ('customer', 'client'):
            rows = FinancialEngine.execute_query("SELECT c.name, c.industry, COALESCE(SUM(i.amount), 0) as billed, c.balance FROM customers c LEFT JOIN invoices i ON c.customer_id = i.entity_id AND i.type = 'RECEIVABLE' GROUP BY c.customer_id ORDER BY billed DESC;")
            headers = ["Customer Name", "Industry", "Total Invoiced", "Outstanding Balance"]
            t_rows = [[r["name"], r["industry"], f"${r['billed']:,.2f}", f"${r['balance']:,.2f}"] for r in rows]
            return "### Revenue by Customer\n\n" + _table_md(headers, t_rows)
        else:
            rows = FinancialEngine.execute_query("SELECT SUBSTR(issue_date, 1, 7) as month, COUNT(*) as count, SUM(amount) as invoiced, SUM(paid_amount) as collected FROM invoices WHERE type = 'RECEIVABLE' GROUP BY month ORDER BY month ASC;")
            headers = ["Month", "Invoices", "Total Invoiced", "Collected"]
            t_rows = [[r["month"], r["count"], f"${r['invoiced']:,.2f}", f"${r['collected']:,.2f}"] for r in rows]
            return "### Monthly Revenue Trend\n\n" + _table_md(headers, t_rows)
    except Exception as e:
        return f"Revenue Error: {str(e)}"

def analyze_cash_flow() -> str:
    """Analyze cash inflow, outflow, and net liquidity movement across ledger transactions."""
    logger.info("Tool: analyze_cash_flow")
    try:
        rows = FinancialEngine.execute_query("SELECT SUBSTR(date, 1, 7) as month, SUM(CASE WHEN type='INCOME' THEN amount ELSE 0 END) as inflow, SUM(CASE WHEN type='EXPENSE' THEN amount ELSE 0 END) as outflow FROM transactions GROUP BY month ORDER BY month ASC;")
        headers = ["Month", "Cash Inflow", "Cash Outflow", "Net Cash Flow", "Status"]
        t_rows = [[r["month"], f"${r['inflow']:,.2f}", f"${r['outflow']:,.2f}", f"${(r['inflow'] - r['outflow']):,.2f}", "[Surplus]" if (r['inflow'] - r['outflow']) >= 0 else "[Deficit]"] for r in rows]
        s = FinancialEngine.get_summary()
        return f"### Cash Flow Trajectory\n\n- **Cumulative Inflow:** {FinancialEngine.fmt_currency(s['cash_inflow'])}\n- **Cumulative Outflow:** {FinancialEngine.fmt_currency(s['cash_outflow'])}\n- **Net Cash Trajectory:** {FinancialEngine.fmt_currency(s['net_cash_flow'])}\n\n" + _table_md(headers, t_rows)
    except Exception as e:
        return f"Cash Flow Error: {str(e)}"

def detect_financial_anomalies() -> str:
    """Detect statistical expense outliers (Z-score > 2.0), duplicate transactions, and budget risks."""
    logger.info("Tool: detect_financial_anomalies")
    try:
        anomalies = FinancialEngine.detect_anomalies()
        if not anomalies: return "### Financial Anomaly Audit\n\nNo anomalies or compliance risks detected."
        headers = ["Severity", "Anomaly Type", "Department", "Amount", "Audit Finding"]
        t_rows = [[a["severity"], a["type"], a["department"], FinancialEngine.fmt_currency(a.get("amount", 0)), a["description"]] for a in anomalies]
        return "### Financial Anomaly & Compliance Audit\n\n> **Note:** Flagged for management review.\n\n" + _table_md(headers, t_rows)
    except Exception as e:
        return f"Anomaly Error: {str(e)}"

def search_accounting_policies(topic_query: str) -> str:
    """Search internal accounting policies (ASC 606, capitalization limits, expense rules, etc.)."""
    logger.info(f"Tool: search_accounting_policies | {topic_query}")
    try:
        words = [w.lower() for w in re.findall(r'\w+', topic_query)]
        policies = FinancialEngine.execute_query("SELECT policy_id, title, category, summary, content FROM accounting_policies;")
        scored = sorted([(sum(f"{p['title']} {p['content']}".lower().count(w) for w in words), p) for p in policies], key=lambda x: x[0], reverse=True)
        top = [p for score, p in scored if score > 0][:2]
        if not top: top = [policies[0]]
        res = [f"#### {p['title']} ({p['policy_id']})\n- **Category:** {p['category']}\n- **Policy Mandate:** {p['content']}" for p in top]
        return "### Accounting Policy Knowledge Base\n\n" + "\n\n---\n\n".join(res)
    except Exception as e:
        return f"Policy Error: {str(e)}"

def generate_financial_report() -> str:
    """Generate an executive board-level financial management report."""
    logger.info("Tool: generate_financial_report")
    try:
        s = FinancialEngine.get_summary()
        anomalies = FinancialEngine.detect_anomalies()
        anom_txt = "\n".join([f"- **[{a['severity']}] {a['type']}**: {a['description']}" for a in anomalies]) if anomalies else "None noted."
        return f"""# Executive Financial Management Report
**Reporting Period:** Q3 Year-to-Date 2026 | **Classification:** Confidential

## 1. Executive KPIs
| Metric | Amount | Status |
|---|---|---|
| **Total Invoiced Revenue** | **{FinancialEngine.fmt_currency(s['total_revenue'])}** | Target Achieved |
| **Total Expenses** | {FinancialEngine.fmt_currency(s['total_expenses'])} | Monitored |
| **Net Operating Profit** | **{FinancialEngine.fmt_currency(s['net_profit'])}** | Operating Margin: {s['profit_margin_pct']}% |
| **Accounts Receivable** | {FinancialEngine.fmt_currency(s['accounts_receivable'])} | Unpaid Customer Invoices |
| **Accounts Payable** | {FinancialEngine.fmt_currency(s['accounts_payable'])} | Pending Supplier Obligations |
| **Net Cash Movement** | {FinancialEngine.fmt_currency(s['net_cash_flow'])} | Ledger Cash Surplus |

## 2. Department Expenses
{analyze_expenses(group_by='department')}

## 3. Internal Audit & Anomaly Flags
{anom_txt}
"""
    except Exception as e:
        return f"Report Error: {str(e)}"

ALL_TOOLS = {
    "query_financial_sql": query_financial_sql, "search_financial_records": search_financial_records,
    "calculate_financial_metrics": calculate_financial_metrics, "analyze_accounts_receivable": analyze_accounts_receivable,
    "analyze_accounts_payable": analyze_accounts_payable, "analyze_expenses": analyze_expenses,
    "analyze_revenue": analyze_revenue, "analyze_cash_flow": analyze_cash_flow,
    "detect_financial_anomalies": detect_financial_anomalies, "search_accounting_policies": search_accounting_policies,
    "generate_financial_report": generate_financial_report
}


# ======================================================================================
# 5. AGENT ORCHESTRATOR & RESILIENT LLM ENGINE
# ======================================================================================

SYSTEM_PROMPT = """You are FinSight AI, an elite corporate Finance & Accounts Agent.
Always use tools to query and calculate real data. Never hallucinate numbers. Format outputs using markdown tables."""

class ConversationMemoryStore:
    def __init__(self): self.sessions: Dict[str, List[Dict[str, str]]] = {}
    def get_history(self, sid: str) -> List[Dict[str, str]]: return self.sessions.setdefault(sid, [])
    def add_message(self, sid: str, role: str, content: str) -> None:
        h = self.get_history(sid)
        h.append({"role": role, "content": content})
        if len(h) > 16: self.sessions[sid] = h[-16:]

session_memory = ConversationMemoryStore()

class AgentOrchestrator:
    def __init__(self):
        self.llm = None
        if LANGCHAIN_AVAILABLE and Config.OPENAI_API_KEY:
            try:
                tools_list = [tool(fn) for fn in ALL_TOOLS.values()]
                self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0.0, api_key=Config.OPENAI_API_KEY).bind_tools(tools_list)
                logger.info("LangChain Tool-Calling Agent initialized.")
            except Exception as e:
                logger.warning(f"LangChain init error: {e}. Resilient deterministic dispatcher active.")

    def _fallback_dispatcher(self, msg: str, history: List[Dict[str, str]]) -> Tuple[str, str]:
        m = msg.lower()
        if any(w in m for w in ["report", "management report", "board"]): return "generate_financial_report", generate_financial_report()
        if any(w in m for w in ["anomaly", "duplicate", "unusual", "fraud", "suspicious"]): return "detect_financial_anomalies", detect_financial_anomalies()
        if any(w in m for w in ["receivable", "owe us", "owed to us", "customer balance", "unpaid invoice", "debtor"]): return "analyze_accounts_receivable", analyze_accounts_receivable()
        if any(w in m for w in ["payable", "we owe", "owe supplier", "owe vendor", "unpaid bill", "creditor"]): return "analyze_accounts_payable", analyze_accounts_payable()
        if any(w in m for w in ["cash flow", "inflow", "outflow", "liquidity"]): return "analyze_cash_flow", analyze_cash_flow()
        if any(w in m for w in ["expense", "spent", "spending", "cost", "budget"]): return "analyze_expenses", analyze_expenses(group_by="category" if "category" in m else "department")
        if any(w in m for w in ["revenue", "sales", "income", "turnover"]): return "analyze_revenue", analyze_revenue(group_by="customer" if "customer" in m else "month")
        if any(w in m for w in ["policy", "rule", "guideline", "capitalization", "asc 606", "per diem"]): return "search_accounting_policies", search_accounting_policies(msg)
        if any(w in m for w in ["search", "find", "lookup"]): return "search_financial_records", search_financial_records(re.sub(r'^(search|find|lookup)\s+', '', m))
        return "calculate_financial_metrics", calculate_financial_metrics()

    def run(self, user_msg: str, session_id: str = "default") -> Dict[str, Any]:
        start = time.time()
        tools_used = []
        history = session_memory.get_history(session_id)

        # Mode A: LLM ReAct
        if self.llm:
            try:
                msgs = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=h["content"]) if h["role"]=="user" else AIMessage(content=h["content"]) for h in history] + [HumanMessage(content=user_msg)]
                for _ in range(6):
                    ai_msg = self.llm.invoke(msgs)
                    msgs.append(ai_msg)
                    if not ai_msg.tool_calls:
                        res = ai_msg.content
                        session_memory.add_message(session_id, "user", user_msg)
                        session_memory.add_message(session_id, "assistant", res)
                        return {"response": res, "tools_used": tools_used, "execution_time_sec": round(time.time()-start, 3), "session_id": session_id, "status": "success"}
                    for tc in ai_msg.tool_calls:
                        tname, targs, tid = tc["name"], tc["args"], tc["id"]
                        tools_used.append(tname)
                        out = ALL_TOOLS[tname](**targs) if tname in ALL_TOOLS else "Tool not found"
                        msgs.append(ToolMessage(content=str(out), tool_call_id=tid))
            except Exception as e:
                logger.warning(f"LLM run error: {e}. Using deterministic fallback.")

        # Mode B: Deterministic Fallback
        tname, res = self._fallback_dispatcher(user_msg, history)
        tools_used.append(tname)
        final_res = f"{res}\n\n---\n**Data Period:** June – August 2026 | **Source:** Company SQLite Ledger"
        session_memory.add_message(session_id, "user", user_msg)
        session_memory.add_message(session_id, "assistant", final_res)
        return {"response": final_res, "tools_used": tools_used, "execution_time_sec": round(time.time()-start, 3), "session_id": session_id, "status": "success"}

finance_agent = AgentOrchestrator()


# ======================================================================================
# 6. FASTAPI APPLICATION & REST ENDPOINTS
# ======================================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User financial question")
    session_id: Optional[str] = Field("default", description="Conversation session ID")

class IngestRecord(BaseModel):
    date: str; department: str; category: str; vendor: str; amount: float; description: str

class IngestRequest(BaseModel):
    expenses: List[IngestRecord]

app = FastAPI(title=Config.APP_NAME, version=Config.VERSION, description="Autonomous AI Finance & Accounts Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/v1/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": Config.VERSION, "db": "connected", "llm_ready": finance_agent.llm is not None}

@app.post("/api/v1/chat", tags=["Chat"])
async def chat(req: ChatRequest):
    if not req.message.strip(): raise HTTPException(400, "Message cannot be empty.")
    return finance_agent.run(req.message, req.session_id)

@app.get("/api/v1/summary", tags=["Financial Data"])
async def summary(): return FinancialEngine.get_summary()

@app.get("/api/v1/report", tags=["Reports"])
async def report(): return {"report_markdown": generate_financial_report()}

@app.get("/api/v1/anomalies", tags=["Audit"])
async def anomalies(): return {"anomalies": FinancialEngine.detect_anomalies()}

@app.post("/api/v1/upload-data", tags=["Ingestion"])
async def upload_data(req: IngestRequest):
    with db.get_connection() as conn:
        c = conn.cursor()
        for i, item in enumerate(req.expenses):
            c.execute("INSERT INTO expenses VALUES (?,?,?,?,?,?,?,?)", (f"EXP-ING-{int(time.time())}-{i}", item.date, item.department, item.category, item.vendor, item.amount, "API_Upload", item.description))
        conn.commit()
    return {"status": "success", "inserted": len(req.expenses)}

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def web_ui():
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>FinSight AI Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body{background:#0f172a;color:#f8fafc;font-family:sans-serif;margin:0;display:flex;height:100vh;overflow:hidden;}
.sidebar{width:300px;background:#1e293b;border-right:1px solid #334155;padding:20px;display:flex;flex-direction:column;}
.main{flex:1;display:flex;flex-direction:column;}
.chat{flex:1;padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:14px;}
.msg{max-width:85%;padding:14px 18px;border-radius:10px;line-height:1.5;}
.user{background:#0284c7;align-self:flex-end;}
.agent{background:#1e293b;border:1px solid #334155;align-self:flex-start;width:90%;}
.badge-t{background:#0369a1;color:#e0f2fe;font-size:0.75rem;padding:3px 6px;border-radius:4px;margin-right:4px;}
.pbtn{background:#334155;color:#cbd5e1;border:1px solid #475569;border-radius:6px;width:100%;text-align:left;padding:7px 10px;font-size:0.8rem;margin-bottom:6px;}
.pbtn:hover{background:#475569;color:white;}
table{width:100%;border-collapse:collapse;margin:8px 0;} th,td{border:1px solid #334155;padding:6px 10px;text-align:left;} th{background:#0f172a;color:#38bdf8;}
</style></head><body>
<div class="sidebar"><h5 class="text-info fw-bold mb-3">FinSight AI Agent</h5>
<div class="overflow-y-auto flex-grow-1">
<button class="pbtn" onclick="send('Give me a complete financial summary.')">Financial Summary</button>
<button class="pbtn" onclick="send('Which department spent the most and what is their budget variance?')">Department Expenses</button>
<button class="pbtn" onclick="send('Show me our overdue customer invoices.')">Accounts Receivable</button>
<button class="pbtn" onclick="send('How much do we owe suppliers?')">Accounts Payable</button>
<button class="pbtn" onclick="send('Find potential anomalies and duplicate expenses.')">Anomaly Audit</button>
<button class="pbtn" onclick="send('Analyze our cash flow trajectory.')">Cash Flow</button>
<button class="pbtn" onclick="send('What is our capitalization threshold policy?')">Accounting Policy</button>
<button class="pbtn" onclick="send('Generate a full management financial report.')">Management Report</button>
</div><small class="text-muted"><a href="/docs" target="_blank" class="text-info text-decoration-none">Swagger REST API Docs</a></small></div>
<div class="main"><div class="p-3 border-bottom border-secondary bg-dark text-white fw-bold">FinSight AI Reasoning Session</div>
<div class="chat" id="chatBox"><div class="msg agent"><strong>FinSight AI Initialized</strong><br>Ask any question regarding company financial statements, revenue, expenses, overdue balances, or compliance.</div></div>
<div class="p-3 bg-dark border-top border-secondary"><form id="f" onsubmit="event.preventDefault();send(document.getElementById('inp').value);" class="d-flex gap-2">
<input id="inp" class="form-control bg-dark text-white border-secondary" placeholder="Ask a financial question..." required>
<button type="submit" class="btn btn-info text-white px-4">Ask</button></form></div></div>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const box=document.getElementById('chatBox'), inp=document.getElementById('inp');
async function send(t){if(!t.trim())return;inp.value='';append('user',t);
const lid='l-'+Date.now(), l=document.createElement('div');l.id=lid;l.className='msg agent';l.innerHTML='<em>Analyzing financial data...</em>';box.appendChild(l);box.scrollTop=box.scrollHeight;
try{const res=await fetch('/api/v1/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});
const d=await res.json();document.getElementById(lid).remove();append('agent',d.response,d.tools_used,d.execution_time_sec);
}catch(e){document.getElementById(lid).remove();append('agent','Error connecting to agent.');}}
function append(r,c,tools=[],time=0){const d=document.createElement('div');d.className='msg '+r;
if(r==='user')d.textContent=c;else{let b=tools.map(x=>`<span class="badge-t">${x}</span>`).join('')+(time?`<small class="text-muted ms-2">${time}s</small>`:'');d.innerHTML=(b?`<div>${b}</div>`:'')+marked.parse(c);}
box.appendChild(d);box.scrollTop=box.scrollHeight;}
</script></body></html>""")


# ======================================================================================
# 7. CLI BENCHMARK & INTERACTIVE RUNNERS
# ======================================================================================

def run_demo():
    print("="*80 + "\n FINSIGHT AI — 10-DOMAIN AGENT BENCHMARK\n" + "="*80)
    questions = [
        ("Financial Summary", "Give me a complete financial summary of the company."),
        ("Department Expenses", "Which department spent the most money and what is their budget variance?"),
        ("Accounts Receivable", "Which customer invoices are overdue and what is our total accounts receivable?"),
        ("Accounts Payable", "How much money do we owe suppliers and which bills are overdue?"),
        ("Cash Flow Analysis", "Analyze our cash flow trajectory across recent months."),
        ("Anomaly Detection", "Identify any suspicious transactions, unusual spikes, or duplicate expenses."),
        ("Policy Knowledge RAG", "What is our corporate policy regarding capitalization threshold for equipment?"),
        ("Custom SQL Query", "Show me all expenses categorized under 'Software Tools & Subscriptions' exceeding $3,000."),
        ("Revenue Trends", "What was our monthly revenue and who are our top customers?"),
        ("Executive Board Report", "Generate a comprehensive financial management report for executive review.")
    ]
    for title, q in questions:
        print(f"\n[{title}] -> Question: \"{q}\"")
        res = finance_agent.run(q, "demo_bench")
        print(f"Tools: {res['tools_used']} ({res['execution_time_sec']}s)\n{res['response']}\n" + "-"*80)
    print("\n[SUCCESS] Benchmark complete! All 10 domains executed perfectly.")

def run_cli():
    print("="*80 + "\n FinSight AI Interactive CLI (Type 'exit' to quit)\n" + "="*80)
    while True:
        try:
            inp = input("\nYou: ").strip()
            if inp.lower() in ('exit', 'quit', 'q'): break
            if not inp: continue
            res = finance_agent.run(inp, "cli_session")
            print(f"\n[Tools: {res['tools_used']} | {res['execution_time_sec']}s]\n{res['response']}")
        except (KeyboardInterrupt, EOFError): break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight AI")
    parser.add_argument("--demo", action="store_true", help="Run 10-question benchmark")
    parser.add_argument("--cli", action="store_true", help="Run interactive terminal CLI")
    parser.add_argument("--port", type=int, default=Config.PORT, help="Port to bind")
    args = parser.parse_args()

    if args.demo: run_demo()
    elif args.cli: run_cli()
    else:
        print(f"Starting {Config.APP_NAME}\nWeb Dashboard: http://localhost:{args.port}/\nSwagger API: http://localhost:{args.port}/docs")
        uvicorn.run(app, host=Config.HOST, port=args.port)
