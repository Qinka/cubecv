// 该文件是 Shanan CV 项目的一部分。
// src/postprocess/detection.rs - 针对目标检测算法的后处理相关功能
//
// 本文件根据 Apache 许可证第 2.0 版（以下简称“许可证”）授权使用；
// 除非遵守该许可证条款，否则您不得使用本文件。
// 您可通过以下网址获取许可证副本：
// http://www.apache.org/licenses/LICENSE-2.0
// 除非适用法律要求或书面同意，根据本许可协议分发的软件均按“原样”提供，
// 不附带任何形式的明示或暗示的保证或条件。
// 有关许可权限与限制的具体条款，请参阅本许可协议。
//
// Copyright (C) 2026 Johann Li <me@qinka.pro>, Wareless Group

mod yolo26_sbc;
pub use yolo26_sbc::{Yolo26Sbc, Yolo26SbcConfig, Yolo26SbcError};
