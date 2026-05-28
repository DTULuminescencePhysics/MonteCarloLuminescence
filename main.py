from src.main_function import parse_launcher_args, hydra_main, hydra_save, run_prepared_experiment

if __name__ == "__main__":
    args = parse_launcher_args()
    if args.run is not None:
        run_prepared_experiment(args.run)
    else:
        if args.save: 
            hydra_save()
        else:
            hydra_main()
 
