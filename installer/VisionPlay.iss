; Inno Setup script for the Windows installer: VisionPlay-Kurulum.exe
;
; Built by .github/workflows/windows.yml after PyInstaller has made
; dist\VisionPlay\. By hand, on Windows, from the project folder:
;
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\VisionPlay.iss
;
; It installs for the current user only, into their own AppData, so it needs
; no administrator rights -- the computers at a school or a fair seldom give
; them out.

#ifndef AppVersion
  #define AppVersion "1.0"
#endif

[Setup]
; never change: Windows recognises a new version of the same program by it
AppId={{01178866-2A8D-4502-A325-F5C938AC2006}
AppName=VisionPlay
AppVersion={#AppVersion}
AppVerName=VisionPlay {#AppVersion}
AppPublisher=VisionPlay
AppPublisherURL=https://github.com/Alp-Che/VisionPlay
DefaultDirName={localappdata}\Programs\VisionPlay
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=VisionPlay-Kurulum
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\VisionPlay.exe
UninstallDisplayName=VisionPlay
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; a game left running would hold its own files open
CloseApplications=yes

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; the whole PyInstaller folder: VisionPlay.exe needs _internal next to it
Source: "..\dist\VisionPlay\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; an older install's leftovers, so files a new version no longer ships do not linger
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\VisionPlay"; Filename: "{app}\VisionPlay.exe"
Name: "{autodesktop}\VisionPlay"; Filename: "{app}\VisionPlay.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VisionPlay.exe"; Description: "{cm:LaunchProgram,VisionPlay}"; Flags: nowait postinstall skipifsilent
