
/*
    Steganography Detection Rules
*/

rule Steganography_Tools
{
    meta:
        description = "Detects steganography tool signatures"
        severity = "medium"
        
    strings:
        $steg1 = "steghide" nocase
        $steg2 = "outguess" nocase
        $steg3 = "jsteg" nocase
        $steg4 = "f5stego" nocase
        
    condition:
        any of them
}

rule Hidden_Archive_Signature
{
    meta:
        description = "Detects hidden archive files"
        severity = "low"
        
    strings:
        $zip = { 50 4B 03 04 }  // ZIP signature
        $rar = { 52 61 72 21 1A 07 00 }  // RAR signature
        $7z = { 37 7A BC AF 27 1C }  // 7Z signature
        
    condition:
        any of them
}
            