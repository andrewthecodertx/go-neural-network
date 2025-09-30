# Stage 1: Build the application
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go.mod and go.sum files to download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the application
# -ldflags="-w -s" strips debug information, making the binary smaller
# CGO_ENABLED=0 is important for a static binary
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o go-neuralnetwork .

# Stage 2: Create the final, minimal image
FROM alpine:latest

WORKDIR /root/

# Copy the built binary from the builder stage
COPY --from=builder /app/go-neuralnetwork .

# Copy data and models (optional, but good for a default run)
COPY redwinequality.csv .
COPY saved_models/ ./saved_models/

# Set the entrypoint
ENTRYPOINT ["./go-neuralnetwork"]
