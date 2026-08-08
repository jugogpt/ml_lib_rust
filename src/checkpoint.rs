use crate::model::{ModelContext, MV_FLAG_PARAMETER};
use crate::optim::Adam;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"MLCK";
const VERSION: u32 = 1;

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    for &x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32_slice(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    for x in &mut out {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        *x = f32::from_le_bytes(b);
    }
    Ok(out)
}

/// Save parameter tensors (and optional Adam moments) to a native checkpoint.
pub fn save_checkpoint(
    path: impl AsRef<Path>,
    model: &ModelContext,
    adam: Option<&Adam>,
) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(MAGIC)?;
    write_u32(&mut f, VERSION)?;

    let params = model.parameter_indices();
    write_u32(&mut f, params.len() as u32)?;

    for &i in &params {
        let val = model.vars[i].val();
        write_u64(&mut f, i as u64)?;
        write_u32(&mut f, val.ndim() as u32)?;
        for &d in &val.shape {
            write_u64(&mut f, d as u64)?;
        }
        write_u32(&mut f, val.numel() as u32)?;
        write_f32_slice(&mut f, &val.data)?;
    }

    match adam {
        Some(opt) => {
            write_u32(&mut f, 1)?; // has optimizer
            let (step, slots) = opt.export_state();
            write_u64(&mut f, step)?;
            write_u32(&mut f, slots.len() as u32)?;
            for (k, m, v) in slots {
                write_u64(&mut f, k as u64)?;
                write_u32(&mut f, m.len() as u32)?;
                write_f32_slice(&mut f, &m)?;
                write_f32_slice(&mut f, &v)?;
            }
            // hyperparams for resume
            write_f32_slice(&mut f, &[opt.beta1, opt.beta2, opt.eps, opt.weight_decay])?;
        }
        None => {
            write_u32(&mut f, 0)?;
        }
    }

    Ok(())
}

/// Load parameters into an existing compiled model (indices/shapes must match).
pub fn load_checkpoint(
    path: impl AsRef<Path>,
    model: &ModelContext,
    adam: Option<&mut Adam>,
) -> io::Result<()> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not an MLCK checkpoint",
        ));
    }
    let ver = read_u32(&mut f)?;
    if ver != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version {ver}"),
        ));
    }

    let nparams = read_u32(&mut f)? as usize;
    let expected = model.parameter_indices();
    if nparams != expected.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "param count mismatch: file {nparams}, model {}",
                expected.len()
            ),
        ));
    }

    for _ in 0..nparams {
        let idx = read_u64(&mut f)? as usize;
        if model.vars.get(idx).map(|v| v.flags & MV_FLAG_PARAMETER == 0) != Some(false) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint index {idx} is not a parameter"),
            ));
        }
        let ndim = read_u32(&mut f)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(&mut f)? as usize);
        }
        let numel = read_u32(&mut f)? as usize;
        let data = read_f32_slice(&mut f, numel)?;
        let mut val = model.vars[idx].val_mut();
        if val.shape != shape || val.numel() != numel {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shape mismatch at param {idx}: file {shape:?}, model {:?}",
                    val.shape
                ),
            ));
        }
        val.data.copy_from_slice(&data);
    }

    let has_opt = read_u32(&mut f)?;
    if has_opt == 1 {
        let step = read_u64(&mut f)?;
        let nslots = read_u32(&mut f)? as usize;
        let mut slots = Vec::with_capacity(nslots);
        for _ in 0..nslots {
            let k = read_u64(&mut f)? as usize;
            let n = read_u32(&mut f)? as usize;
            let m = read_f32_slice(&mut f, n)?;
            let v = read_f32_slice(&mut f, n)?;
            slots.push((k, m, v));
        }
        let hypers = read_f32_slice(&mut f, 4)?;
        if let Some(opt) = adam {
            opt.beta1 = hypers[0];
            opt.beta2 = hypers[1];
            opt.eps = hypers[2];
            opt.weight_decay = hypers[3];
            opt.import_state(step, slots);
        }
    }

    Ok(())
}
