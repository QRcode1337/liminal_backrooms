# Liminal Backrooms iOS

Native SwiftUI MVP for the Liminal Backrooms conversation loop.

## What is included

- SwiftUI tab shell for conversation, agents, and settings.
- OmniRoute chat-completions client (OpenAI-compatible local router).
- Editable seed prompt, turn count, agent names, model IDs, and personas.
- Local transcript export through the iOS share sheet.

The existing Python/PyQt desktop app is unchanged. This is a first native iOS target, not an embedded Python runtime.

## Build

Open `LiminalBackrooms.xcodeproj` in Xcode, or build from the repository root:

```sh
xcodebuild \
  -project ios/LiminalBackrooms/LiminalBackrooms.xcodeproj \
  -target LiminalBackrooms \
  -sdk iphonesimulator \
  -configuration Debug \
  SYMROOT=/tmp/liminal-ios-build \
  OBJROOT=/tmp/liminal-ios-obj \
  build
```

Using `/tmp` for build products avoids file-provider extended attributes that can make simulator code signing fail in synced folders.
