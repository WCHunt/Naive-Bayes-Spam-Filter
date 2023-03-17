BIN_NAME=classifier
MAIN_FILE=classifier.go
BUILD_DIR=build
SRC_FILES=$(wildcard *.go)

# Targets
.PHONY: all clean

all: build

build: $(SRC_FILES)
	@echo "Building $(BIN_NAME)"
	@mkdir -p $(BUILD_DIR)
	@go build -o $(BUILD_DIR)/$(BIN_NAME) $(MAIN_FILE)

clean:
	@echo "Cleaning up"
	@rm -rf $(BUILD_DIR)