# Semantic Image Search Engine 🔍

A modern, fast, and efficient semantic image search engine built with FastAPI, PyTorch, and LanceDB. This application allows users to search for similar images using either image queries through an elegant web interface.

![Demo Screenshot Placeholder]

## Features ✨

- **Drag & Drop Interface**: Simple and intuitive image upload
- **Real-time Search**: Fast similarity search using state-of-the-art models
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Metadata Display**: View detailed information about each image
- **Latin Name Generator**: Generate Latin descriptions for matched images
- **Vector Database**: Efficient similarity search using LanceDB
- **Multiple Model Support**: Configurable with different backbone models

## Tech Stack 🛠️

- **Backend**: FastAPI, PyTorch, timm
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Database**: LanceDB
- **Models**: ResNet50, MobileNetV3 (configurable)
- **Image Processing**: Pillow, scikit-learn

## Installation 🚀

1. Clone the repository:

```bash
git clone https://github.com/yourusername/image-search-engine.git
cd image-search-engine
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your settings:

- Copy `configs/ocean_resnet50.yml.example` to `configs/ocean_resnet50.yml`
- Update the configuration with your settings:
  ```yaml
  COLLECTION_NAME: your_collection_name
  LANCEDB: "./db/your_db.lance"
  MODEL_NAME: "resnet50.a1_in1k"
  MODEL_DIM: 2048
  DATASET_PATH: "./your_dataset_path"
  ```

## Usage 💡

1. Ingest your images into the database:

```bash
python ingest.py
```

2. Start the server:

```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:8000`

## Project Structure 📁

```
image-search-engine/
├── app.py              # FastAPI application
├── ingest.py           # Image ingestion script
├── encoder.py          # Feature extraction
├── db_utils.py         # Database utilities
├── configs/            # Configuration files
├── static/             # Web interface files
│   ├── index.html
│   └── style.css
└── dataset/           # Your image dataset
```

## Configuration ⚙️

The application supports multiple configuration files in the `configs/` directory:

- `ocean_resnet50.yml`: ResNet50 configuration
- `animals_resnet50.yml`: Alternative ResNet50 configuration
- `animals_mobilenetv3.yml`: MobileNetV3 configuration

Each configuration file specifies:

- Model architecture and dimensions
- Database location and collection name
- Dataset path

## API Endpoints 🌐

- `GET /`: Web interface
- `POST /search/image`: Image similarity search
  - Parameters:
    - `file`: Image file (multipart/form-data)
    - `limit`: Number of results (default: 25)

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [LanceDB](https://github.com/lancedb/lancedb)
- [timm](https://github.com/rwightman/pytorch-image-models)
