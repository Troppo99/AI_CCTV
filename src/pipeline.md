::: mermaid
graph TD;
    A[Start] --> B[Initialize AICCTV, REPORT, SAVER]
    B --> C[Start Frame Capture Thread]
    C --> D[Read Frame from Queue]
    D --> E{Frame None?}
    E -- Yes --> F[Log Error]
    E -- No --> G[Resize Mask]
    G --> H[Process Frame]
    H --> I[Update Data Table]
    I --> J[Calculate Percentages]
    J --> K[Draw Table]
    K --> L{Save Frame?}
    L -- Yes --> M[Write Frame to Video]
    L -- No --> N
    M --> N{Send to SQL?}
    N -- Yes --> O[Send Data to SQL]
    N -- No --> Q
    O --> Q
    Q{Check for Interrupt or Termination}
    Q -- Yes --> R[End]
    Q -- No --> D
:::