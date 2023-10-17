import qrcode
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Function to generate the QR code and save it as an image file
def generate_qr_code(url, filename):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)

# Function to create a PDF with the QR code and user-entered name
def create_pdf_with_qr_code(url, user_name, pdf_filename):
    # Generate QR code
    qr_code_filename = "qrcode.png"
    generate_qr_code(url, qr_code_filename)

    # Create PDF
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    story = []

    # Add QR code image
    qr_code = Image(qr_code_filename, width=3*inch, height=3*inch)
    story.append(qr_code)
    
    # Add user name as text
    styles = getSampleStyleSheet()
    user_name_text = Paragraph(f"Name: {user_name}", styles["Normal"])
    story.append(user_name_text)
    
    # Add some spacing
    story.append(Spacer(1, 0.5*inch))

    # Build the PDF document
    doc.build(story)

    # Clean up: remove the QR code image
    import os
    os.remove(qr_code_filename)

if __name__ == "__main__":
    url = input("Enter the URL: ")
    user_name = input("Enter your name: ")
    pdf_filename = input("Enter the PDF filename (e.g., output.pdf): ")

    create_pdf_with_qr_code(url, user_name, pdf_filename)
    print(f"PDF with QR code and name saved as {pdf_filename}")
