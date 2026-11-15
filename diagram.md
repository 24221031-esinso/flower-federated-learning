# Test Mermaid

```mermaid
graph TD
    A[Mulai] --> B[Proses]
    B --> C[Selesai]
    ```mermaid
flowchart TD
    subgraph Aktornya
        A[ServerApp]
        B[ClientApp 1]
        C[ClientApp 2]
        D[ClientApp 3]
        E[Flower Dashboard]
        F[Operator]
    end

    subgraph UseCase
        UC1((Inisialisasi Model Global))
        UC2((Distribusi Parameter))
        UC3((Training Lokal CIFAR-10))
        UC4((Pengiriman Hasil))
        UC5((Aggregasi FedAvg))
        UC6((Pemantauan Metrik))
        UC7((Pengaturan Ronde & Klien))
    end

    A --> UC1
    A --> UC2
    A --> UC5
    A --> UC7

    B --> UC3
    B --> UC4
    C --> UC3
    C --> UC4
    D --> UC3
    D --> UC4

    E --> UC6
    F --> UC7