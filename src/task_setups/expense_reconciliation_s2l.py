"""Workspace setup for the expense_reconciliation_s2l task."""

import json
import random
from pathlib import Path

import pandas as pd


FINANCE_DIR = Path("Documents/Finance")
EXPENSES_DOC_PATH = FINANCE_DIR / "expenses.xlsx"
CATEGORY_RULES_PATH = FINANCE_DIR / "category_rules.txt"
INVOICES_DOC_PATH = FINANCE_DIR / "invoices.xlsx"
PAYMENTS_DOC_PATH = FINANCE_DIR / "payments.xlsx"

EXPENSES_OUTPUT_FILENAME = "expenses_corrected.xlsx"
ANALYSIS_OUTPUT_FILENAME = "expense_analysis.csv"
FLAGGED_PAYMENTS_OUTPUT_FILENAME = "flagged_payments.xlsx"

CATEGORIES = ["Travel", "Meals", "Office Supplies", "Software", "Training", "Entertainment"]

# Keywords are checked in category priority order (Travel first, Entertainment last).
# Within each category, the first matching keyword wins.
CATEGORY_KEYWORDS = {
    "Travel": ["flight", "hotel", "motel", "taxi", "uber", "lyft", "train ticket", "parking", "car rental", "airfare", "shuttle"],
    "Meals": ["restaurant", "cafe", "coffee", "lunch", "dinner", "breakfast", "catering", "snack", "food"],
    "Office Supplies": ["printer", "paper", "pens", "stapler", "notebooks", "binder", "toner", "desk supplies", "pencils"],
    "Software": ["software", "saas", "subscription", "license", "cloud service", "app"],
    "Training": ["online course", "training seminar", "workshop", "conference", "book purchase", "certification", "webinar"],
    "Entertainment": ["team outing", "movie", "concert", "tickets", "team event", "game", "recreation", "celebration"],
}

CATEGORY_RULES_TEXT = """Expense Categorization Rules
=============================

Assign each expense to the FIRST matching category below (case-insensitive keyword search
on the Description field). Apply rules in the listed order.

1. Travel
   Keywords: flight, hotel, motel, taxi, uber, lyft, train ticket, parking, car rental, airfare, shuttle

2. Meals
   Keywords: restaurant, cafe, coffee, lunch, dinner, breakfast, catering, snack, food

3. Office Supplies
   Keywords: printer, paper, pens, stapler, notebooks, binder, toner, desk supplies, pencils

4. Software
   Keywords: software, saas, subscription, license, cloud service, app

5. Training
   Keywords: online course, training seminar, workshop, conference, book purchase, certification, webinar

6. Entertainment
   Keywords: team outing, movie, concert, tickets, team event, game, recreation, celebration
"""

DESCRIPTIONS_BY_CATEGORY = {
    "Travel": [
        "Flight to New York",
        "Hotel stay in Chicago",
        "Taxi to airport",
        "Uber to client site",
        "Train ticket to Boston",
        "Parking fee at office",
        "Car rental for business trip",
        "Airfare to Seattle",
        "Shuttle to downtown hotel",
        "Lyft to morning meeting",
    ],
    "Meals": [
        "Restaurant dinner with client",
        "Cafe breakfast before meeting",
        "Coffee for team",
        "Team lunch order",
        "Client dinner downtown",
        "Breakfast meeting",
        "Catering for all-hands",
        "Snack for team",
        "Food delivery for late session",
    ],
    "Office Supplies": [
        "Printer cartridges (set of 4)",
        "Paper reams for office",
        "Pens and markers bundle",
        "Stapler and staples",
        "Notebooks (pack of 10)",
        "Binder clips and folders",
        "Toner cartridge replacement",
        "Desk supplies assortment",
        "Pencils and erasers pack",
    ],
    "Software": [
        "Software license renewal",
        "SaaS subscription monthly",
        "Cloud service fee",
        "App subscription annual",
        "Subscription to project tool",
        "License for design software",
    ],
    "Training": [
        "Online course - Python programming",
        "Training seminar on leadership",
        "Workshop attendance fee",
        "Conference registration fee",
        "Book purchase - Data Science",
        "Certification exam fee",
        "Webinar on architecture",
    ],
    "Entertainment": [
        "Team outing - bowling night",
        "Movie tickets for team",
        "Concert tickets for client",
        "Tickets to industry game",
        "Team event - escape room",
        "Recreation center booking",
        "Team celebration party",
    ],
}

EMPLOYEE_POOL = [
    "Alice Chen", "Bob Martinez", "Carol Johnson", "David Kim", "Emma Wilson",
    "Frank Lee", "Grace Zhang", "Henry Nguyen", "Iris Patel", "James Brown",
    "Karen Davis", "Leo Rodriguez", "Maria Garcia", "Nathan Scott", "Olivia Taylor",
    "Peter Wang", "Quinn Foster", "Rachel Adams", "Sam Thompson", "Tina Hernandez",
]

DEPARTMENT_POOL = ["Finance", "Engineering", "Sales", "HR", "Operations"]

VENDOR_POOL = [
    "Acme Corp", "Brightline Tech", "Crestview Solutions", "Delta Services",
    "Echo Systems", "Frontier Logistics", "Global Dynamics", "Harbor Networks",
    "Infinity Consulting", "Jade Analytics", "Keystone Partners", "Lighthouse Media",
]

TASK_INSTRUCTION = f"""The workspace contains financial documents in `{FINANCE_DIR.as_posix()}`.

Files provided:
- `expenses.xlsx`: Company expense records (columns: Expense_ID, Date, Employee, Department, Description, Amount, Category)
- `category_rules.txt`: Rules for correctly categorizing expenses by description keyword
- `invoices.xlsx`: Vendor invoices (columns: Invoice_ID, Vendor, Description, Amount)
- `payments.xlsx`: Payment records (columns: Payment_ID, Date, Reference, Amount)

Complete all three tasks below:

**Task 1 – Expense Category Correction**
Read `category_rules.txt` and apply its rules to every expense row. Create `{EXPENSES_OUTPUT_FILENAME}` in the workspace root. It must be identical to `expenses.xlsx` except for an added column `Correct_Category` containing the rule-based correct category for every row (including rows already correctly categorized).

**Task 2 – Expense Analysis Report**
Using the corrected categories from Task 1, create `{ANALYSIS_OUTPUT_FILENAME}` in the workspace root. It must be a CSV with exactly these columns:
- `Category`: the expense category
- `Total_Amount`: total Amount across all expenses in that category (rounded to 2 decimal places)
- `Number_of_Employees`: count of distinct employees with at least one expense in that category
- `Cost_Per_Employee`: Total_Amount / Number_of_Employees, rounded to 2 decimal places

Only include categories that have at least one expense.

**Task 3 – Invoice / Payment Reconciliation**
Match each payment to its invoice using the `Reference` field (which contains an Invoice_ID). Flag any payment where:
- The referenced Invoice_ID does not exist in `invoices.xlsx` — Issue: "Invoice {{ID}} not found"
- The payment Amount does not match the invoice Amount — Issue: "Amount mismatch for invoice {{ID}}: expected {{expected}}, paid {{paid}}"

Create `{FLAGGED_PAYMENTS_OUTPUT_FILENAME}` in the workspace root with columns `Payment_ID` and `Issue`. In the last row, add a summary: Payment_ID = "TOTAL", Issue = "Total amount mismatch: Invoices={{invoices_total}}, Payments={{payments_total}}" (totals formatted to 2 decimal places).
"""


def classify(description: str) -> str:
    desc_lower = description.lower()
    for category in CATEGORIES:
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in desc_lower:
                return category
    return "Other"


def build_prompt() -> str:
    return TASK_INSTRUCTION


def _build_expenses(rng: random.Random, employees: list[str], departments: dict[str, str]) -> list[dict]:
    rows = []
    expense_id = 1
    for employee in employees:
        num_expenses = rng.randint(2, 5)
        for _ in range(num_expenses):
            true_category = rng.choice(CATEGORIES)
            description = rng.choice(DESCRIPTIONS_BY_CATEGORY[true_category])
            amount = round(rng.uniform(15.0, 500.0), 2)
            month = rng.randint(1, 9)
            day = rng.randint(1, 28)
            date = f"2024-0{month}-{day:02d}"

            # Miscategorize ~25% of expenses
            if rng.random() < 0.25:
                wrong_categories = [c for c in CATEGORIES if c != true_category]
                assigned_category = rng.choice(wrong_categories)
            else:
                assigned_category = true_category

            rows.append({
                "Expense_ID": f"EXP-{expense_id:03d}",
                "Date": date,
                "Employee": employee,
                "Department": departments[employee],
                "Description": description,
                "Amount": amount,
                "Category": assigned_category,
            })
            expense_id += 1
    return rows


def _build_invoices(rng: random.Random, num_invoices: int) -> list[dict]:
    rows = []
    services = ["IT Services", "Office Supplies", "Professional Services", "Software", "Consulting"]
    for i in range(1, num_invoices + 1):
        vendor = rng.choice(VENDOR_POOL)
        service = rng.choice(services)
        amount = round(rng.uniform(200.0, 5000.0), 2)
        rows.append({
            "Invoice_ID": f"INV-{i:03d}",
            "Vendor": vendor,
            "Description": f"{service} - Q1 2024",
            "Amount": amount,
        })
    return rows


def _build_payments(rng: random.Random, invoices: list[dict]) -> tuple[list[dict], list[dict]]:
    payments = []
    issues = []
    payment_id = 1
    invoice_amounts = {inv["Invoice_ID"]: inv["Amount"] for inv in invoices}

    # Normal payments for all but last 2 invoices
    for inv in invoices[:-2]:
        payments.append({
            "Payment_ID": f"PAY-{payment_id:03d}",
            "Date": f"2024-01-{rng.randint(10, 28):02d}",
            "Reference": inv["Invoice_ID"],
            "Amount": inv["Amount"],
        })
        payment_id += 1

    # Anomaly 1: reference to non-existent invoice
    fake_id = "INV-999"
    bad_ref_amount = round(rng.uniform(100.0, 500.0), 2)
    payments.append({
        "Payment_ID": f"PAY-{payment_id:03d}",
        "Date": f"2024-01-{rng.randint(10, 28):02d}",
        "Reference": fake_id,
        "Amount": bad_ref_amount,
    })
    issues.append({
        "Payment_ID": f"PAY-{payment_id:03d}",
        "Issue": f"Invoice {fake_id} not found",
    })
    payment_id += 1

    # Anomaly 2: amount mismatch for second-to-last invoice
    inv = invoices[-2]
    delta = rng.choice([50.0, 100.0, -50.0])
    wrong_amount = round(inv["Amount"] + delta, 2)
    payments.append({
        "Payment_ID": f"PAY-{payment_id:03d}",
        "Date": f"2024-01-{rng.randint(10, 28):02d}",
        "Reference": inv["Invoice_ID"],
        "Amount": wrong_amount,
    })
    issues.append({
        "Payment_ID": f"PAY-{payment_id:03d}",
        "Issue": f"Amount mismatch for invoice {inv['Invoice_ID']}: expected {inv['Amount']:.2f}, paid {wrong_amount:.2f}",
    })
    payment_id += 1

    # Normal payment for last invoice
    inv = invoices[-1]
    payments.append({
        "Payment_ID": f"PAY-{payment_id:03d}",
        "Date": f"2024-01-{rng.randint(10, 28):02d}",
        "Reference": inv["Invoice_ID"],
        "Amount": inv["Amount"],
    })
    payment_id += 1

    total_invoices = round(sum(inv["Amount"] for inv in invoices), 2)
    total_payments = round(sum(p["Amount"] for p in payments), 2)
    issues.append({
        "Payment_ID": "TOTAL",
        "Issue": f"Total amount mismatch: Invoices={total_invoices:.2f}, Payments={total_payments:.2f}",
    })

    return payments, issues


def build_expected_expenses_corrected(expenses: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(expenses)
    df["Correct_Category"] = df["Description"].apply(classify)
    return df


def build_expected_analysis(expenses_corrected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in sorted(CATEGORIES):
        cat_df = expenses_corrected_df[expenses_corrected_df["Correct_Category"] == category]
        if len(cat_df) == 0:
            continue
        total_amount = round(cat_df["Amount"].sum(), 2)
        num_employees = int(cat_df["Employee"].nunique())
        cost_per_employee = round(total_amount / num_employees, 2)
        rows.append({
            "Category": category,
            "Total_Amount": total_amount,
            "Number_of_Employees": num_employees,
            "Cost_Per_Employee": cost_per_employee,
        })
    return pd.DataFrame(rows)


def build_example(index: int, seed: int) -> dict:
    rng = random.Random(seed)

    num_employees = rng.randint(6, 12)
    num_invoices = rng.randint(8, 12)

    employees = rng.sample(EMPLOYEE_POOL, num_employees)
    departments = {emp: rng.choice(DEPARTMENT_POOL) for emp in employees}

    expenses = _build_expenses(rng, employees, departments)
    invoices = _build_invoices(rng, num_invoices)
    payments, issues = _build_payments(rng, invoices)

    return {
        "id": f"expense_reconciliation_{index}",
        "seed": seed,
        "employees": employees,
        "departments": departments,
        "expenses": expenses,
        "invoices": invoices,
        "payments": payments,
        "issues": issues,
        "prompt": build_prompt(),
    }


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    finance_dir = workspace / FINANCE_DIR
    finance_dir.mkdir(parents=True, exist_ok=True)

    expenses_df = pd.DataFrame(example["expenses"])
    invoices_df = pd.DataFrame(example["invoices"])
    payments_df = pd.DataFrame(example["payments"])

    expenses_df.to_excel(workspace / EXPENSES_DOC_PATH, index=False)
    invoices_df.to_excel(workspace / INVOICES_DOC_PATH, index=False)
    payments_df.to_excel(workspace / PAYMENTS_DOC_PATH, index=False)
    (workspace / CATEGORY_RULES_PATH).write_text(CATEGORY_RULES_TEXT, encoding="utf-8")

    groundtruth_dir = Path(log_dir) / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)

    expenses_corrected_df = build_expected_expenses_corrected(example["expenses"])
    analysis_df = build_expected_analysis(expenses_corrected_df)
    issues_df = pd.DataFrame(example["issues"])

    expenses_corrected_df.to_excel(groundtruth_dir / "expenses_corrected.xlsx", index=False)
    analysis_df.to_csv(groundtruth_dir / "expense_analysis.csv", index=False)
    issues_df.to_excel(groundtruth_dir / "flagged_payments.xlsx", index=False)

    (groundtruth_dir / "example.json").write_text(
        json.dumps(example, indent=2), encoding="utf-8"
    )
