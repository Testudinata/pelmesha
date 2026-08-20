```mermaid
flowchart TD
    subgraph ORCH["DataSet orchestrator (serving.py)"]
        SET["DataSet<br/>(entry point)"]
        ADD["add_sources<br/>add one or more sources on demand<br/>path | list | dir | {path: config}<br/>+ config & kde_configs"]
        RREF["set_reference_source"]
    end

    subgraph LOAD["Loading & metadata (filling.py)"]
        DM["DataManager.get_loader<br/>selects loader by extension"]
        subgraph LOADERS["Loader classes (BaseLoader subclasses)"]
            L1["loader_imzml"]
            L2["loader_cdf"]
            L3["Custom loader"]
        end
        BL["BaseLoader (ABC)<br/>create_metafile<br/>→ ingredients.hdf5"]
        MD["Per-ROI metadata:<br/>spectra, coords, ROI bounds,<br/>m/z discretization (discret_coeffs)"]
        DS["DataSource<br/>(composition: delegates to loader)"]
    end

    subgraph PREP["Preparation (cookbook.py)"]
        PDS["PreparedDataSource<br/>set_link(datasource)"]
        CFG["PipelineConfigurator → Configs<br/>per-ROI configs with defaults<br/>(preprocess/process/peakpick)"]
    end

    subgraph PIPELINE["Pipeline (serving.py)"]
        PIP["Pipeline<br/>_multistream_pipeline"]
        subgraph MAINPATH["Main path to feature matrix"]
            P2["peakpick()<br/>→ peaklists.hdf5"]
            P3["estimate_peak_density_kde()<br/>uses discret_coeffs in KDE<br/>→ peaks_density.hdf5"]
        end
        subgraph BRANCH["Branch: processed spectra"]
            P1["process()<br/>→ processed_spectra.hdf5"]
            USER["Custom user processing<br/>(e.g. data binning)"]
        end
    end

    subgraph REF["Reference peaks (optional)"]
        RP["get_reference_peaks<br/>(KDE + m/z correction)"]
        AP["set_align_peaks_from_ref<br/>writes align_peaks & align_pweights<br/>into Configs"]
    end

    subgraph FEAT["Feature matrix"]
        FM["feature_matrix<br/>(Pipeline / DataSet)"]
        KM["m/z correction:<br/>_summerize_kde_mz +<br/>apply_kde_mzcorrection"]
        FILT["filter by frequency +<br/>consensus peaks"]
        OUT["RESULT:<br/>feature_matrix<br/>pivot + save .parquet"]
    end

    %% Entry point: user calls DataSet, then adds sources on demand
    SET --> ADD

    %% Each added source is loaded + prepared
    ADD --> DM
    DM --> LOADERS
    LOADERS --> BL
    BL --> MD
    MD --> DS
    DS --> PDS
    CFG --> PDS
    PDS --> SET

    %% DataSet drives the pipeline
    SET --> PIP

    %% Optional reference source
    SET --> RREF
    RREF --> RP
    RP --> AP
    AP --> CFG

    %% Branch: processed spectra (separate path, not to feature matrix)
    PIP -.->|separate call| P1
    P1 --> USER

    %% Main path to feature matrix
    PIP --> P2
    P2 --> P3
    P3 --> KM
    P2 --> FM
    FM --> KM
    KM --> FILT
    FILT --> OUT
```