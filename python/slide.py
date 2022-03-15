import PyPDF2 as pdf
reader = pdf.PdfFileReader(open('need.pdf', 'rb'))
print(type(reader.pages), reader.numPages)
page = reader.getPage(0)
writer = pdf.PdfFileWriter()
writer.addPage(page)
with open('1.pdf', 'wb') as f:
    writer.write(f)