:::mermaid
  graph TD;
      A[Start] --> B[Initialize AICCTV & REPORT]
      B --> C{Server Mode?}
      C -->|Yes| D[Get Server Configuration]
      C -->|No| E
      D --> E{Save Video?}
      E -->|Yes| F[Initialize Video Saver]
      E -->|No| G
      F --> G[Load Mask if Exists]
      G --> H[Start Capture Thread]
      H --> I[Main Loop]
      I --> J{Frame Available in Queue?}
      J -->|Yes| K[Get Frame from Queue]
      K --> L{Frame is None?}
      L -->|Yes| M[Print Warning & Continue]
      L -->|No| N[Resize Mask]
      N --> O[Process Frame with YOLO]
      O --> P[Update Employee Activity Data]
      P --> Q[Draw Detection Info on Frame]
      Q --> R{Save Video?}
      R -->|Yes| S[Write Frame to Video Saver]
      R -->|No| T
      S --> T
      T --> U[Resize Frame for Display]
      U --> V{Show Frame?}
      V -->|Yes| W[Display Frame with OpenCV]
      V -->|No| X
      W --> X
      X --> Y{Send Data to SQL?}
      Y -->|Yes| Z[Send Data to SQL]
      Y -->|No| AA
      Z --> AA[Check for Exit Key]
      AA --> I
      AA --> AB[Exit]
      AB --> AC[Close Video Writer if Open]
      AC --> AD[Release Video Capture]
      AD --> AE[End]
:::