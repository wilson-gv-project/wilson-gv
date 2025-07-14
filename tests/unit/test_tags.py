
def test_tags() -> None:
    print("\n")
    from wilson.utils.tagger import TAG_REGISTRY

    for name, tags in TAG_REGISTRY.items():
        print(f"   {name}:".ljust(38)+f" {tags}")
