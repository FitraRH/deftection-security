
/*
    Polyglot File Detection Rules
*/

rule Polyglot_PDF_Image
{
    meta:
        description = "Detects PDF/Image polyglot files"
        severity = "high"
        
    strings:
        $pdf_header = "%PDF"
        $jpeg_header = { FF D8 FF }
        $png_header = { 89 50 4E 47 }
        
    condition:
        $pdf_header and ($jpeg_header or $png_header)
}

rule Suspicious_Metadata_Length
{
    meta:
        description = "Detects unusually large metadata sections"
        severity = "medium"
        
    condition:
        filesize > 1MB 
}
            