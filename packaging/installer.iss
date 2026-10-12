; Inno Setup - packs the dist\Luboo folder into a single setup.exe. Run via packaging\build.ps1.
; All paths are relative to the repo root (SourceDir).
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
; "-PRIVATE" when built with build.ps1 -EmbedKey (build with an embedded Groq key - only send to trusted people)
#ifndef Suffix
  #define Suffix ""
#endif

[Setup]
SourceDir=..
AppId={{8F2C4D7A-3B1E-4C9A-9E52-6A0D1F7B3C21}
AppName=Luboo
AppVersion={#AppVersion}
AppPublisher=Luboo
DefaultDirName={autopf}\Luboo
DefaultGroupName=Luboo
; Install for the current user only - no Administrator rights needed
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=dist
OutputBaseFilename=Luboo-Setup-{#AppVersion}{#Suffix}
SetupIconFile=resources\icon.ico
UninstallDisplayIcon={app}\Luboo.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=THIRD_PARTY_NOTICES.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\Luboo\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "THIRD_PARTY_NOTICES.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Luboo"; Filename: "{app}\Luboo.exe"
Name: "{group}\{cm:UninstallProgram,Luboo}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Luboo"; Filename: "{app}\Luboo.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Luboo.exe"; Description: "{cm:LaunchProgram,Luboo}"; Flags: nowait postinstall skipifsilent
