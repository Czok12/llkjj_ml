#!/bin/bash
# LLKJJ ML Pipeline - Production Deployment Script (Native)
# Usage: ./deploy-production.sh [environment]

set -e  # Exit on any error

# Configuration
ENVIRONMENTS=("dev" "staging" "prod")
PYTHON_VERSION="3.12"
PROJECT_NAME="llkjj-ml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check Poetry
    if ! command -v poetry &> /dev/null; then
        log_warning "Poetry not found. Installing..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        exit 1
    fi

    log_success "All requirements satisfied"
}

setup_environment() {
    local env=$1
    log_info "Setting up environment: $env"

    # Create environment directory
    ENV_DIR="/opt/${PROJECT_NAME}/${env}"
    sudo mkdir -p "$ENV_DIR"
    sudo chown "$USER:$USER" "$ENV_DIR"
    cd "$ENV_DIR"

    # Clone or update repository
    if [ -d "${PROJECT_NAME}" ]; then
        log_info "Updating existing repository..."
        cd "${PROJECT_NAME}"
        git pull origin main
    else
        log_info "Cloning repository..."
        git clone https://github.com/Czok12/llkjj_ml.git "${PROJECT_NAME}"
        cd "${PROJECT_NAME}"
    fi

    # Install dependencies
    log_info "Installing dependencies with Poetry..."
    poetry env use python3
    poetry install --no-dev --no-interaction

    # Create environment-specific configuration
    create_env_config "$env"

    log_success "Environment $env setup complete"
}

create_env_config() {
    local env=$1
    log_info "Creating environment configuration for $env..."

    # Create .env file based on environment
    case $env in
        "dev")
            cat > .env << EOF
# Development Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_DEBUG=true
GOOGLE_API_KEY=\${GOOGLE_API_KEY_DEV}
# Add development-specific settings
EOF
            ;;
        "staging")
            cat > .env << EOF
# Staging Environment Configuration
ENVIRONMENT=staging
LOG_LEVEL=INFO
ENABLE_DEBUG=false
GOOGLE_API_KEY=\${GOOGLE_API_KEY_STAGING}
# Add staging-specific settings
EOF
            ;;
        "prod")
            cat > .env << EOF
# Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=WARNING
ENABLE_DEBUG=false
GOOGLE_API_KEY=\${GOOGLE_API_KEY_PROD}
# Add production-specific settings
EOF
            ;;
    esac

    log_success "Environment configuration created"
}

install_system_service() {
    local env=$1
    log_info "Installing system service for $env..."

    # Create systemd service file
    sudo tee "/etc/systemd/system/${PROJECT_NAME}-${env}.service" > /dev/null << EOF
[Unit]
Description=LLKJJ ML Pipeline - $env
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/${PROJECT_NAME}/${env}/${PROJECT_NAME}
Environment=PATH=/opt/${PROJECT_NAME}/${env}/${PROJECT_NAME}/.venv/bin
ExecStart=/opt/${PROJECT_NAME}/${env}/${PROJECT_NAME}/.venv/bin/python main.py --environment=$env
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable "${PROJECT_NAME}-${env}.service"

    log_success "System service installed and enabled"
}

run_health_check() {
    local env=$1
    log_info "Running health check for $env..."

    cd "/opt/${PROJECT_NAME}/${env}/${PROJECT_NAME}"

    # Run tests
    log_info "Running test suite..."
    poetry run pytest tests/ -x --tb=short

    # Security audit
    log_info "Running security audit..."
    poetry run python main.py security-audit

    # Performance check
    log_info "Running performance test..."
    poetry run pytest tests/test_pipeline_e2e.py::TestPipelinePerformance::test_single_document_performance -v

    log_success "Health check passed"
}

# Main deployment logic
main() {
    local environment=${1:-"dev"}

    # Validate environment
    if [[ ! " ${ENVIRONMENTS[@]} " =~ " ${environment} " ]]; then
        log_error "Invalid environment: $environment"
        log_info "Valid environments: ${ENVIRONMENTS[*]}"
        exit 1
    fi

    log_info "🚀 Starting LLKJJ ML Pipeline deployment for: $environment"

    # Step 1: Check requirements
    check_requirements

    # Step 2: Setup environment
    setup_environment "$environment"

    # Step 3: Install system service (for prod/staging)
    if [[ "$environment" != "dev" ]]; then
        install_system_service "$environment"
    fi

    # Step 4: Run health check
    run_health_check "$environment"

    # Step 5: Start service (for prod/staging)
    if [[ "$environment" != "dev" ]]; then
        log_info "Starting service..."
        sudo systemctl start "${PROJECT_NAME}-${environment}.service"
        sudo systemctl status "${PROJECT_NAME}-${environment}.service"
    fi

    log_success "🎉 Deployment completed successfully!"
    log_info "Environment: $environment"
    log_info "Location: /opt/${PROJECT_NAME}/${environment}/${PROJECT_NAME}"

    if [[ "$environment" != "dev" ]]; then
        log_info "Service: ${PROJECT_NAME}-${environment}.service"
        log_info "Logs: sudo journalctl -u ${PROJECT_NAME}-${environment}.service -f"
    fi
}

# Run main function with all arguments
main "$@"
