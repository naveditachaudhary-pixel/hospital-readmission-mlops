import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.makedirs("docs", exist_ok=True)

WIDTH, HEIGHT = 900, 560
BG_COLOR = (248, 250, 252) # #F8FAFC
CARD_BG = (255, 255, 255)
BORDER_COLOR = (226, 232, 240)
TEXT_DARK = (30, 41, 59)
TEXT_MUTED = (100, 116, 139)
ACCENT_BLUE = (37, 99, 235)
HIGH_RISK_BG = (253, 232, 232)
HIGH_RISK_TXT = (155, 28, 28)
MOD_RISK_BG = (254, 240, 138)
MOD_RISK_TXT = (133, 77, 14)
LOW_RISK_BG = (222, 247, 236)
LOW_RISK_TXT = (3, 84, 63)

def get_font(size=14, bold=False):
    font_paths = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                index = 1 if bold and p.endswith(".ttc") else 0
                return ImageFont.truetype(p, size=size, index=index)
            except Exception:
                pass
    return ImageFont.load_default()

font_title = get_font(22, bold=True)
font_sub = get_font(13)
font_card_val = get_font(20, bold=True)
font_label = get_font(12, bold=True)
font_body = get_font(12)
font_small = get_font(11)

def draw_header(draw):
    # App Title & Subtitle
    draw.text((30, 24), "🏥 Hospital Readmission Risk Predictor", font=font_title, fill=TEXT_DARK)
    draw.text((30, 56), "MLOps Pipeline Clinical Intelligence & Real-Time Patient Risk Assessment", font=font_sub, fill=TEXT_MUTED)
    draw.line([(30, 80), (WIDTH - 30, 80)], fill=BORDER_COLOR, width=1)

def draw_card(draw, x, y, w, h, title, val, badge=None, badge_bg=None, badge_txt=None):
    draw.rectangle([x, y, x + w, y + h], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((x + 16, y + 14), title, font=font_label, fill=TEXT_MUTED)
    draw.text((x + 16, y + 38), val, font=font_card_val, fill=TEXT_DARK)
    if badge and badge_bg and badge_txt:
        # Draw pill badge
        bx, by = x + 16, y + 72
        draw.rounded_rectangle([bx, by, bx + 120, by + 24], radius=12, fill=badge_bg)
        draw.text((bx + 16, by + 4), badge, font=font_small, fill=badge_txt)

def create_frame(step=1):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(draw)

    # Input Mode Section
    draw.rectangle([30, 96, WIDTH - 30, 140], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((45, 108), "Input Mode:", font=font_label, fill=TEXT_DARK)
    
    # Radio options
    draw.ellipse([135, 112, 147, 124], outline=ACCENT_BLUE if step>=1 else BORDER_COLOR, width=2)
    if step >= 1:
        draw.ellipse([138, 115, 144, 121], fill=ACCENT_BLUE)
    draw.text((155, 108), "🎲 Random Patient Sample", font=font_body, fill=TEXT_DARK)

    draw.ellipse([345, 112, 357, 124], outline=BORDER_COLOR, width=2)
    draw.text((365, 108), "🎛️ Interactive Clinical Builder", font=font_body, fill=TEXT_MUTED)

    # Patient Data Table Frame
    draw.rectangle([30, 152, WIDTH - 30, 240], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((45, 162), "Patient Record #1042 — Demographics & Clinical Factors", font=font_label, fill=TEXT_DARK)
    
    headers = ["Age Group", "Hospital Days", "Lab Tests", "Medications", "Prior Emergencies", "Prior Inpatient"]
    vals = ["60-70 yrs", "5 Days", "64 Procedures", "18 Prescriptions", "2 Visits", "1 Admission"]
    
    col_w = (WIDTH - 90) // len(headers)
    for i, (h, v) in enumerate(zip(headers, vals)):
        cx = 45 + i * col_w
        draw.text((cx, 188), h, font=font_small, fill=TEXT_MUTED)
        draw.text((cx, 208), v, font=font_body, fill=TEXT_DARK)

    if step >= 2:
        # Results Metric Cards
        draw_card(draw, 30, 252, 195, 108, "READMISSION LIKELIHOOD", "68.4%", "HIGH RISK", HIGH_RISK_BG, HIGH_RISK_TXT)
        draw_card(draw, 240, 252, 195, 108, "MODEL DECISION", "Readmit <30d")
        draw_card(draw, 450, 252, 195, 108, "GROUND TRUTH", "Readmitted", "✅ Correct Match", LOW_RISK_BG, LOW_RISK_TXT)
        draw_card(draw, 660, 252, 210, 108, "CHAMPION MODEL", "HistGradientBoosting")

    if step >= 3:
        # Protocol Box
        draw.rectangle([30, 372, 420, 535], fill=(239, 246, 255), outline=(191, 219, 254), width=1)
        draw.rectangle([30, 372, 35, 535], fill=ACCENT_BLUE) # left border accent
        draw.text((50, 384), "🩺 Recommended Post-Discharge Care Protocol", font=font_label, fill=ACCENT_BLUE)
        
        protocols = [
            "• Mandatory 48-Hour Phone Check-in post discharge",
            "• Fast-track primary care / endocrinology visit in 7d",
            "• Complete pharmacy diabetes medication reconciliation",
            "• Provide daily glucose monitoring instructions"
        ]
        for idx, p in enumerate(protocols):
            draw.text((50, 415 + idx * 28), p, font=font_body, fill=TEXT_DARK)

    if step >= 4:
        # SHAP Plot Frame
        draw.rectangle([435, 372, WIDTH - 30, 535], fill=CARD_BG, outline=BORDER_COLOR, width=1)
        draw.text((450, 384), "🔍 Local SHAP Decision Attribution", font=font_label, fill=TEXT_DARK)
        
        # Horizontal bars simulating SHAP waterfall values
        features = ["inpatient_ratio (+0.24)", "number_emergency (+0.18)", "num_medications (+0.12)", "is_high_risk_discharge (+0.09)", "age (+0.05)"]
        bar_widths = [190, 140, 95, 70, 40]
        
        for idx, (feat, bw) in enumerate(zip(features, bar_widths)):
            fy = 412 + idx * 23
            draw.text((450, fy), feat, font=font_small, fill=TEXT_MUTED)
            draw.rectangle([630, fy + 3, 630 + bw, fy + 13], fill=(239, 68, 68)) # Red positive impact

    return img

def build_gif():
    print("Generating demo GIF frames...")
    frames = []
    
    # Step 1: Initial load
    f1 = create_frame(step=1)
    frames.extend([f1] * 8)
    
    # Step 2: Prediction calculated
    f2 = create_frame(step=2)
    frames.extend([f2] * 10)
    
    # Step 3: Protocol generated
    f3 = create_frame(step=3)
    frames.extend([f3] * 10)
    
    # Step 4: SHAP plot generated
    f4 = create_frame(step=4)
    frames.extend([f4] * 20)
    
    out_path = "docs/demo.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0
    )
    print(f"Successfully created {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == "__main__":
    build_gif()
