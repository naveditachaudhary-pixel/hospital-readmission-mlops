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

# Load logo if available
LOGO = None
if os.path.exists("assets/logo.png"):
    try:
        LOGO = Image.open("assets/logo.png").convert("RGBA").resize((44, 44))
    except Exception:
        pass

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

font_title = get_font(20, bold=True)
font_sub = get_font(12)
font_card_val = get_font(18, bold=True)
font_label = get_font(12, bold=True)
font_body = get_font(11)
font_small = get_font(11)

def draw_header(img, draw):
    tx = 30
    if LOGO:
        img.paste(LOGO, (30, 20), LOGO)
        tx = 84

    # App Title & Subtitle
    draw.text((tx, 20), "Hospital Readmission Risk Predictor", font=font_title, fill=TEXT_DARK)
    draw.text((tx, 48), "MLOps Pipeline Clinical Intelligence & Real-Time Risk Assessment", font=font_sub, fill=TEXT_MUTED)
    draw.line([(30, 76), (WIDTH - 30, 76)], fill=BORDER_COLOR, width=1)

def draw_card(draw, x, y, w, h, title, val, badge=None, badge_bg=None, badge_txt=None):
    draw.rectangle([x, y, x + w, y + h], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((x + 16, y + 14), title, font=font_label, fill=TEXT_MUTED)
    draw.text((x + 16, y + 36), val, font=font_card_val, fill=TEXT_DARK)
    if badge and badge_bg and badge_txt:
        bx, by = x + 16, y + 68
        draw.rounded_rectangle([bx, by, bx + 110, by + 22], radius=10, fill=badge_bg)
        draw.text((bx + 14, by + 4), badge, font=font_small, fill=badge_txt)

def create_frame(step=1):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_header(img, draw)

    # Input Mode Section
    draw.rectangle([30, 90, WIDTH - 30, 134], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((45, 103), "Patient Source:", font=font_label, fill=TEXT_DARK)
    
    # Radio options
    draw.ellipse([145, 106, 157, 118], outline=ACCENT_BLUE if step>=1 else BORDER_COLOR, width=2)
    if step >= 1:
        draw.ellipse([148, 109, 154, 115], fill=ACCENT_BLUE)
    draw.text((165, 103), "Sample Real Patient from Dataset", font=font_body, fill=TEXT_DARK)

    draw.ellipse([375, 106, 387, 118], outline=BORDER_COLOR, width=2)
    draw.text((395, 103), "Customize Patient Profile", font=font_body, fill=TEXT_MUTED)

    # Patient Data Table Frame
    draw.rectangle([30, 146, WIDTH - 30, 234], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    draw.text((45, 156), "Clinical Profile — Patient #1042", font=font_label, fill=TEXT_DARK)
    
    headers = ["Age Group", "Hospital Stay", "Lab Tests", "Medications", "Prior Emergencies", "Prior Inpatient"]
    vals = ["60-70 yrs", "5 Days", "64 Procedures", "18 Prescriptions", "2 Visits", "1 Admission"]
    
    col_w = (WIDTH - 90) // len(headers)
    for i, (h, v) in enumerate(zip(headers, vals)):
        cx = 45 + i * col_w
        draw.text((cx, 182), h, font=font_small, fill=TEXT_MUTED)
        draw.text((cx, 202), v, font=font_body, fill=TEXT_DARK)

    if step >= 2:
        # Results Metric Cards
        draw_card(draw, 30, 246, 195, 104, "READMISSION LIKELIHOOD", "68.4%", "HIGH RISK", HIGH_RISK_BG, HIGH_RISK_TXT)
        draw_card(draw, 240, 246, 195, 104, "PREDICTION", "Readmit <30d")
        draw_card(draw, 450, 246, 195, 104, "GROUND TRUTH", "Readmitted", "Match", LOW_RISK_BG, LOW_RISK_TXT)
        draw_card(draw, 660, 246, 210, 104, "ACTIVE MODEL", "HistGradientBoosting")

    if step >= 3:
        # Protocol Box
        draw.rectangle([30, 362, 420, 525], fill=(240, 249, 255), outline=(186, 230, 253), width=1)
        draw.rectangle([30, 362, 35, 525], fill=ACCENT_BLUE)
        draw.text((50, 374), "Recommended Clinical Protocol", font=font_label, fill=ACCENT_BLUE)
        
        protocols = [
            "• Mandatory 48-Hour Nurse Phone Call post discharge",
            "• Fast-track primary care / endocrinology visit in 7d",
            "• Complete pharmacy diabetes medication reconciliation",
            "• Provide daily glucose monitoring instructions"
        ]
        for idx, p in enumerate(protocols):
            draw.text((50, 404 + idx * 27), p, font=font_body, fill=TEXT_DARK)

    if step >= 4:
        # Drivers Plot Frame
        draw.rectangle([435, 362, WIDTH - 30, 525], fill=CARD_BG, outline=BORDER_COLOR, width=1)
        draw.text((450, 374), "Key Clinical Drivers (Risk Impact)", font=font_label, fill=TEXT_DARK)
        
        features = ["Prior Inpatient Stays (+0.24)", "Emergency Visits (+0.18)", "Prescribed Meds (+0.12)", "Discharge Destination (+0.09)", "Age Group (+0.05)"]
        bar_widths = [180, 130, 90, 65, 35]
        
        for idx, (feat, bw) in enumerate(zip(features, bar_widths)):
            fy = 402 + idx * 22
            draw.text((450, fy), feat, font=font_small, fill=TEXT_MUTED)
            draw.rectangle([640, fy + 3, 640 + bw, fy + 12], fill=(239, 68, 68))

    return img

def build_gif():
    print("Generating clean demo GIF frames...")
    frames = []
    
    f1 = create_frame(step=1)
    frames.extend([f1] * 8)
    
    f2 = create_frame(step=2)
    frames.extend([f2] * 10)
    
    f3 = create_frame(step=3)
    frames.extend([f3] * 10)
    
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
