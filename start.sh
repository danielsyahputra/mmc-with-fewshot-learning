if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found."
    exit 1
fi

jupyter lab --allow-root --port ${JUPYTER_PORT} --ip "0.0.0.0"