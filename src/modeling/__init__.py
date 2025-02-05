from .transformer import TransformerModel

__all__ = ['TransformerModel']
def build_model(config):
    return TransformerModel(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    )