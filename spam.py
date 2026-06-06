import pywhatkit

# Send a WhatsApp message
# Phone number must include country code
phone_number = "+918708432559"
message = "APP MERE HOON, KYA AAP MUJHE SPAM KARNA CHAHTE HO? 😂"

# Send instantly after opening WhatsApp Web
for i in range(100):
    pywhatkit.sendwhatmsg_instantly(
        phone_no=phone_number,
        message=message,
        wait_time=15,
        tab_close=True
    )

print("Message sent.")