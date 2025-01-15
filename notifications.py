import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import locale
import os

# Email credentials
MY_ADDRESS = 'noreply@bk.rw'
PASSWORD = '6{8TW*J/'

def send_notification(user_name, user_email, status):
    """
    Send an email notification to the user regarding their approval status.
    """
    # Set locale for proper date formatting
    locale.setlocale(locale.LC_ALL, '')

    # Define file paths
    if status == 'approved.':
        template_file = 'message.html'
    else:
        template_file = 'message_decline.html'
    banner_file = 'img/bkofkgl.png'

    # print('Image path exists',os.path.exists(banner_file))  # Should return True


    # Read message template
    message_template = read_template(template_file)

    # Set up the SMTP server
    s = smtplib.SMTP(host='smtp.office365.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)

    # Create a message
    msg = MIMEMultipart('related')  # 'related' to allow for images in the email

    # Replace placeholders in the template with actual values
    message = message_template.substitute(PERSON_NAME=user_name.title(), STATUS=status)

    # Attach the message body (HTML)
    msg.attach(MIMEText(message, 'html'))

    # Attach the banner image
    with open(banner_file, 'rb') as img_file:
        img = MIMEImage(img_file.read(),_subtype = 'png')
        img.add_header('Content-ID', '<bkofkgl.png>')  # Content-ID for inline image
        img.add_header('Content-Disposition', 'inline', filename=banner_file)
        msg.attach(img)

    # Construct the "To" and "From" fields
    msg['From'] = MY_ADDRESS
    msg['To'] = user_email+'@bk.rw'
    msg['Subject'] = "Your Application Status"

    # Send the email
    try:
        s.send_message(msg)
        print(f"Notification sent successfully to {user_email}")
    except Exception as e:
        print(f"Failed to send notification to {user_email}: {e}")
    finally:
        s.quit()

def read_template(filename):
    """
    Returns a Template object containing the message template from the specified file.
    """
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)