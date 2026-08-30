"""
Generates 3 rich domain-specific PDF documents with structured tables,
headers, and quantitative data for testing Lynx CRAG ingestion and retrieval.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1E1B4B"),
    spaceAfter=12,
)
h2_style = ParagraphStyle(
    "DocH2",
    parent=styles["Heading2"],
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#4338CA"),
    spaceBefore=10,
    spaceAfter=6,
)
body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#1F2937"),
    spaceAfter=8,
)

def create_quantum_pdf():
    pdf_path = DATA_DIR / "quantum_computing_spec.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []

    story.append(Paragraph("Q-Engine 2026: Superconducting Quantum Processor Architecture", title_style))
    story.append(Paragraph("<b>Author:</b> Quantum Hardware & Cryogenics Laboratory | <b>Classification:</b> Proprietary Engineering", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. System Overview and Cryogenic Parameters", h2_style))
    story.append(Paragraph(
        "The Q-Engine 2026 is a 256-qubit superconducting quantum processor utilizing planar transmon architectures. "
        "The dilution refrigerator maintains a base plate operating temperature of <b>14.5 milliKelvin (mK)</b> with a cooling power of 18 uW at base. "
        "Qubit coherence metrics demonstrate a longitudinal relaxation time (T1) of <b>185 microseconds</b> and a dephasing time (T2) of <b>242 microseconds</b>.",
        body_style
    ))

    story.append(Paragraph("2. Gate Fidelity Benchmark Matrix", h2_style))
    table_data = [
        ["Processor Model", "Qubit Count", "Single-Qubit Fidelity", "Two-Qubit (CZ) Fidelity", "Readout Error"],
        ["Q-Engine 128 (Gen 1)", "128 Transmons", "99.92%", "99.45%", "0.85%"],
        ["Q-Engine 256 (Gen 2)", "256 Transmons", "99.98%", "99.86%", "0.32%"],
        ["Q-Engine 512 (Preview)", "512 Transmons", "99.95%", "99.78%", "0.48%"],
    ]
    t = Table(table_data, colWidths=[130, 85, 105, 110, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Quantum Error Mitigation (QEM)", h2_style))
    story.append(Paragraph(
        "Real-time zero-noise extrapolation (ZNE) combined with randomized compiling suppresses coherent control errors by <b>84.2%</b>. "
        "The active surface code cycle time is clocked at <b>200 nanoseconds</b>.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated: {pdf_path}")

def create_clinical_pdf():
    pdf_path = DATA_DIR / "biotech_clinical_trial_q3.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []

    story.append(Paragraph("NeuroShield-7: Phase III Multicenter Clinical Trial Summary", title_style))
    story.append(Paragraph("<b>Sponsor:</b> Aethelgard Therapeutics | <b>Protocol:</b> AG-703-ALZ | <b>Phase:</b> III", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Primary Endpoint Efficacy & Cohort Demographics", h2_style))
    story.append(Paragraph(
        "A randomized, double-blind, placebo-controlled study enrolled <b>2,450 patients</b> across 68 clinical sites. "
        "Patients receiving NeuroShield-7 (10 mg/kg biweekly) demonstrated a statistically significant <b>34.2% slowing of cognitive decline</b> "
        "measured on the Clinical Dementia Rating-Sum of Boxes (CDR-SB, p < 0.001) at 76 weeks.",
        body_style
    ))

    story.append(Paragraph("2. Biomarker Reduction and Safety Profile", h2_style))
    table_data = [
        ["Study Arm", "Patient Count (n)", "CDR-SB Change (76 wks)", "Amyloid PET Reduction", "ARIA-E Incidence"],
        ["Placebo", "1,225", "+1.82 points", "-1.2%", "1.8%"],
        ["NeuroShield-7 Low (5mg)", "612", "+1.38 points (-24.1%)", "-58.6%", "6.4%"],
        ["NeuroShield-7 High (10mg)", "613", "+1.20 points (-34.2%)", "-78.4%", "8.9%"],
    ]
    t = Table(table_data, colWidths=[125, 85, 115, 105, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#059669")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F0FDF4")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1FAE5")),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Regulatory Filing Schedule", h2_style))
    story.append(Paragraph(
        "Biologics License Application (BLA) submission to the US FDA is targeted for <b>February 18, 2027</b> under Priority Review designation.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated: {pdf_path}")

def create_security_pdf():
    pdf_path = DATA_DIR / "cybersecurity_zero_trust_audit.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    story = []

    story.append(Paragraph("Enterprise Zero Trust Architecture & Cryptographic Key Audit", title_style))
    story.append(Paragraph("<b>Auditor:</b> CyberDefense Global | <b>Scope:</b> Tenant Beta Infrastructure", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Post-Quantum Cryptographic Migration", h2_style))
    story.append(Paragraph(
        "All edge ingress gateways and internal service meshes have migrated from RSA-4096 to NIST post-quantum standard algorithms: "
        "<b>ML-KEM (Kyber-768)</b> for key encapsulation and <b>ML-DSA (Dilithium-3)</b> for digital signatures. "
        "Mutual TLS 1.3 ephemeral keys are rotated automatically every <b>48 hours</b>.",
        body_style
    ))

    story.append(Paragraph("2. SOC2 & Zero Trust Compliance Scorecard", h2_style))
    table_data = [
        ["Security Control Domain", "Target Policy", "Current Compliance", "Audit Finding", "Risk Level"],
        ["IAM & MFA Enforcement", "FIDO2 WebAuthn", "100.0%", "Pass (Zero Bypass)", "Low"],
        ["Microsegmentation", "Default Deny East-West", "98.7%", "Pass (1 exception)", "Low"],
        ["SIEM Log Ingestion", "Min 100k EPS", "165,000 EPS", "Pass (Zero drop)", "Low"],
        ["HSM Key Backup", "FIPS 140-3 Level 4", "100.0%", "Pass (Georedundant)", "Low"],
    ]
    t = Table(table_data, colWidths=[120, 100, 95, 115, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DC2626")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#FEF2F2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FEE2E2")),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Incident Response Threshold", h2_style))
    story.append(Paragraph(
        "Mean Time to Detect (MTTD) is <b>42 seconds</b> and Mean Time to Remediate (MTTR) automated network isolation is <b>1.8 seconds</b>.",
        body_style
    ))

    doc.build(story)
    print(f"[OK] Generated: {pdf_path}")

if __name__ == "__main__":
    create_quantum_pdf()
    create_clinical_pdf()
    create_security_pdf()
    print("[SUCCESS] All 3 enterprise PDF test files generated.")
