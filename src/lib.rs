extern crate tch;
extern crate serde;
extern crate serde_json;

use std::{fs, path::Path};
use std::fs::File;
use std::io::BufReader;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use tch::{Device, Kind, Reduction, Tensor, TrainableCModule, kind, vision};
use tch::nn::{VarStore, Adam, Optimizer, OptimizerConfig};

#[derive(Serialize, Deserialize, Debug)]
struct TrainerConfig {
    dataroot: Box<String>,
    modelroot: Box<String>,
    saveroot: Box<String>,
    latent_dim: i64,
    img_size: i64,
    beta1: f64,
    lr: f64,
    batch_size:i64,
    use_gpu:bool,
    max_steps:u32,
    logging_steps:u32,
    eval_steps:u32,
    save_steps:u32,
}

impl TrainerConfig {
    fn new<T: AsRef<Path>>(config_path:T) -> TrainerConfig {
        let file = File::open(config_path).unwrap();
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).unwrap()
    }
}

#[derive(Debug)]
pub struct Trainer {
    config: TrainerConfig,
    discriminator_vs: VarStore,
    generator_vs: VarStore,
    images: Tensor,
    discriminator: TrainableCModule,
    generator: TrainableCModule,
    optimizer_d: Optimizer<Adam>,
    optimizer_g: Optimizer<Adam>,
}

impl Trainer {
    pub fn new<T: AsRef<Path>>(config_path:T) -> Trainer {
        let config = TrainerConfig::new(config_path);
        let device = if config.use_gpu {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };
        let discriminator_vs = VarStore::new(device);
        let generator_vs = VarStore::new(device);

        let images = vision::image::load_dir(
            config.dataroot.as_ref(), config.img_size, config.img_size
        ).expect("failed loading dataset");
        println!("loaded dataset:{:?}", images);

        let discriminator = TrainableCModule::load(
            format!("{}/{}", config.modelroot, "discriminator.pt"), discriminator_vs.root()
        ).expect("failed loading discriminator");
        let generator = TrainableCModule::load(
            format!("{}/{}", config.modelroot, "generator.pt"), generator_vs.root()
        ).expect("failed loading generator");
        let mut optimizer = Adam::default();
        optimizer.beta1 = config.beta1;
        let optimizer_d = optimizer.clone().build(&discriminator_vs, config.lr).unwrap();
        let optimizer_g = optimizer.clone().build(&generator_vs, config.lr).unwrap();

        Trainer {
            config,
            discriminator_vs,
            generator_vs,
            images,
            discriminator,
            generator,
            optimizer_d,
            optimizer_g,
        }
    }

    fn random_batch_images(&self) -> Tensor {
        let index = Tensor::randint(self.images.size()[0], &[self.config.batch_size], kind::INT64_CPU);
        self.images.index_select(0, &index)
            .to(self.discriminator_vs.device())
            .to_kind(Kind::Float)
            / 127.5 - 1.
    }

    fn rand_latent(&self) -> Tensor {
        (Tensor::rand(&[self.config.batch_size, self.config.latent_dim, 1, 1], kind::FLOAT_CPU)*2.0-1.0)
            .to(self.generator_vs.device())
    }

    fn discriminator_step(&mut self) -> f64 {
        self.discriminator_vs.unfreeze();
        self.generator_vs.freeze();
        let discriminator_loss = {
            let batch_images = self.random_batch_images();
            let y_pred = batch_images.apply_t(&self.discriminator, true);
            let y_pred_fake = self.rand_latent()
                .apply_t(&self.generator, true)
                .copy().detach().apply_t(&self.discriminator, true);

            let real_labels = Tensor::ones(y_pred.size().as_ref(), (y_pred.kind(), y_pred.device()));
            let fake_labels = Tensor::zeros(y_pred_fake.size().as_ref(), (y_pred_fake.kind(), y_pred_fake.device()));

            y_pred.binary_cross_entropy::<Tensor>(&real_labels, None, Reduction::Mean)
            + y_pred_fake.binary_cross_entropy::<Tensor>(&fake_labels, None, Reduction::Mean)
        };
        self.optimizer_d.backward_step(&discriminator_loss);
        discriminator_loss.to(Device::Cpu).double_value(&[])
    }

    fn generator_step(&mut self) -> f64 {
        self.discriminator_vs.freeze();
        self.generator_vs.unfreeze();
        let generator_loss = {
            let y_pred_fake = self.rand_latent()
                .apply_t(&self.generator, true)
                .apply_t(&self.discriminator, true);
            let real_labels = Tensor::ones(y_pred_fake.size().as_ref(), (y_pred_fake.kind(), y_pred_fake.device()));
            y_pred_fake.binary_cross_entropy::<Tensor>(&real_labels, None, Reduction::Mean)
        };
        self.optimizer_g.backward_step(&generator_loss);
        generator_loss.to(Device::Cpu).double_value(&[])
    }

    pub fn save(&self, steps:u32) {
        let root = Path::new(self.config.saveroot.as_ref()).join(format!("checkpoint-{}", steps));
        fs::create_dir_all(&root).unwrap();
        self.discriminator.save(root.join("discriminator.pt"))
            .expect("failed save discriminator");
        self.generator.save(root.join("generator.pt"))
            .expect("failed save discriminator");
        println!("saved models");
    }

    pub fn train(&mut self) -> Result<()> {
        let mut discriminator_loss = 0.;
        let mut generator_loss = 0.;
        let fixed_noise = self.rand_latent();

        println!("start train loop");
        for i in 0..self.config.max_steps {
            discriminator_loss += self.discriminator_step();
            generator_loss +=  self.generator_step();
            if i != 0 && i % self.config.eval_steps == 0 {
                let images = fixed_noise.apply_t(&self.generator, false)
                    .view([-1, 3, self.config.img_size, self.config.img_size])
                    .to(Device::Cpu);
                vision::image::save(&image_matrix(&images, 4)?, format!("sample/relout{}.png", i))?
            }

            if i != 0 && i % self.config.logging_steps == 0 {
                discriminator_loss /= f64::from(self.config.logging_steps);
                generator_loss /= f64::from(self.config.logging_steps);
                println!("step {}: d_loss {}, g_loss {}", i, discriminator_loss, generator_loss);
                discriminator_loss = 0.;
                generator_loss = 0.;
            }

            if i != 0 && i % self.config.save_steps == 0 {
                self.save(i);
            }
        }
        Ok(())
    }
}

// Generate a 2D matrix of images from a tensor with multiple images.
// sz × sz に画像を並べて出力
fn image_matrix(imgs: &Tensor, sz: i64) -> Result<Tensor> {
    let imgs = ((imgs + 1.) * 127.5).clamp(0., 255.).to_kind(Kind::Uint8);
    let mut ys: Vec<Tensor> = vec![];
    for i in 0..sz {
        ys.push(Tensor::cat(
            &(0..sz)
                .map(|j| imgs.narrow(0, 4 * i + j, 1))
                .collect::<Vec<_>>(),
            2,
        ))
    }
    Ok(Tensor::cat(&ys, 3).squeeze1(0))
}