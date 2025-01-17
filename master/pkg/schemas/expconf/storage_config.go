package expconf

import (
	"encoding/json"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"github.com/determined-ai/determined/master/pkg/schemas"
	"github.com/determined-ai/determined/master/pkg/union"
)

const (
	// DefaultSharedFSContainerPath is the base storage path inside containers for SharedFS storage.
	DefaultSharedFSContainerPath = "/determined_shared_fs"
	// DefaultSharedFSPropagation is the propagation setting for SharedFS storage.
	DefaultSharedFSPropagation = "rprivate"
)

//go:generate ../gen.sh
// CheckpointStorageConfigV0 has the common checkpoint config params.
type CheckpointStorageConfigV0 struct {
	RawSharedFSConfig *SharedFSConfigV0 `union:"type,shared_fs" json:"-"`
	RawHDFSConfig     *HDFSConfigV0     `union:"type,hdfs" json:"-"`
	RawS3Config       *S3ConfigV0       `union:"type,s3" json:"-"`
	RawGCSConfig      *GCSConfigV0      `union:"type,gcs" json:"-"`
	RawAzureConfig    *AzureConfigV0    `union:"type,azure" json:"-"`

	RawSaveExperimentBest *int `json:"save_experiment_best"`
	RawSaveTrialBest      *int `json:"save_trial_best"`
	RawSaveTrialLatest    *int `json:"save_trial_latest"`
}

// Merge implements schemas.Mergeable.
func (c CheckpointStorageConfigV0) Merge(other interface{}) interface{} {
	return schemas.UnionMerge(c, other)
}

// MarshalJSON implements the json.Marshaler interface.
func (c CheckpointStorageConfigV0) MarshalJSON() ([]byte, error) {
	return union.MarshalEx(c, true)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (c *CheckpointStorageConfigV0) UnmarshalJSON(data []byte) error {
	if err := union.Unmarshal(data, c); err != nil {
		return err
	}
	type DefaultParser *CheckpointStorageConfigV0
	return errors.Wrap(json.Unmarshal(data, DefaultParser(c)), "failed to parse checkpoint storage")
}

// Printable returns a copy the object with secrets hidden.
func (c CheckpointStorageConfigV0) Printable() CheckpointStorageConfigV0 {
	out := schemas.Copy(c).(CheckpointStorageConfigV0)
	hiddenValue := "********"
	if out.RawS3Config != nil {
		if out.RawS3Config.RawAccessKey != nil {
			out.RawS3Config.RawAccessKey = &hiddenValue
		}
		if out.RawS3Config.RawSecretKey != nil {
			out.RawS3Config.RawSecretKey = &hiddenValue
		}
	}
	return out
}

//go:generate ../gen.sh
// TensorboardStorageConfigV0 is a legacy config.
type TensorboardStorageConfigV0 struct {
	RawSharedFSConfigV0 *SharedFSConfigV0 `union:"type,shared_fs" json:"-"`
	RawHDFSConfig       *HDFSConfigV0     `union:"type,hdfs" json:"-"`
	RawS3Config         *S3ConfigV0       `union:"type,s3" json:"-"`
	RawGCSConfig        *GCSConfigV0      `union:"type,gcs" json:"-"`
	RawAzureConfig      *AzureConfigV0    `union:"type,azure" json:"-"`
}

// Merge implements schemas.Mergeable.
func (t TensorboardStorageConfigV0) Merge(other interface{}) interface{} {
	return schemas.UnionMerge(t, other)
}

// MarshalJSON implements the json.Marshaler interface.
func (t TensorboardStorageConfigV0) MarshalJSON() ([]byte, error) {
	return union.Marshal(t)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (t *TensorboardStorageConfigV0) UnmarshalJSON(data []byte) error {
	return union.Unmarshal(data, t)
}

//go:generate ../gen.sh
// SharedFSConfigV0 is a config for shared filesystem storage.
type SharedFSConfigV0 struct {
	RawHostPath        *string `json:"host_path"`
	RawContainerPath   *string `json:"container_path,omitempty"`
	RawCheckpointPath  *string `json:"checkpoint_path,omitempty"`
	RawTensorboardPath *string `json:"tensorboard_path,omitempty"`
	RawStoragePath     *string `json:"storage_path"`
	RawPropagation     *string `json:"propagation"`
}

// PathInContainer caclulates where the full StoragePath will be inside the container.
func (s SharedFSConfigV0) PathInContainer() string {
	if s.RawStoragePath == nil {
		return DefaultSharedFSContainerPath
	}
	if filepath.IsAbs(*s.RawStoragePath) {
		relPath, err := filepath.Rel(*s.RawHostPath, *s.RawStoragePath)
		if err != nil {
			panic("detected unvalidated sharedfs config")
		}
		return filepath.Join(DefaultSharedFSContainerPath, relPath)
	}
	return filepath.Join(DefaultSharedFSContainerPath, *s.RawStoragePath)
}

//go:generate ../gen.sh
// HDFSConfigV0 configures storing checkpoints in HDFS.
type HDFSConfigV0 struct {
	RawURL  *string `json:"hdfs_url"`
	RawPath *string `json:"hdfs_path"`
	RawUser *string `json:"user"`
}

//go:generate ../gen.sh
// S3ConfigV0 configures storing checkpoints on S3.
type S3ConfigV0 struct {
	RawBucket      *string `json:"bucket"`
	RawAccessKey   *string `json:"access_key"`
	RawSecretKey   *string `json:"secret_key"`
	RawEndpointURL *string `json:"endpoint_url"`
	RawPrefix      *string `json:"prefix"`
}

// Validate implements the check.Validatable interface.
func (c S3ConfigV0) Validate() []error {
	var errs []error
	if c.RawPrefix != nil {
		rawPrefix := *c.RawPrefix
		if rawPrefix == ".." || strings.HasPrefix(rawPrefix, "../") ||
			strings.HasSuffix(rawPrefix, "/..") || strings.Contains(rawPrefix, "/../") {
			errs = append(errs, errors.New("'prefix' must not contain /../"))
		}
	}
	return errs
}

//go:generate ../gen.sh
// GCSConfigV0 configures storing checkpoints on GCS.
type GCSConfigV0 struct {
	RawBucket *string `json:"bucket"`
}

// Validate implements the check.Validatable interface.
func (c GCSConfigV0) Validate() []error {
	var errs []error
	if c.RawBucket == nil {
		errs = append(errs, errors.New("'bucket' must be specified"))
	}
	return errs
}

//go:generate ../gen.sh
// AzureConfigV0 configures storing checkpoints on Azure.
type AzureConfigV0 struct {
	RawContainer        *string `json:"container"`
	RawConnectionString *string `json:"connection_string,omitempty"`
	RawAccountURL       *string `json:"account_url,omitempty"`
	RawCredential       *string `json:"credential,omitempty"`
}

// Merge implements schemas.Mergeable.
func (c AzureConfigV0) Merge(other interface{}) interface{} {
	out := AzureConfigV0{}
	otherConfig := other.(AzureConfigV0)
	out.RawContainer = schemas.Merge(c.RawContainer, otherConfig.RawContainer).(*string)
	var credSource AzureConfigV0
	if c.RawConnectionString != nil || c.RawAccountURL != nil {
		credSource = c
	} else {
		credSource = otherConfig
	}
	out.RawConnectionString = schemas.Copy(credSource.RawConnectionString).(*string)
	out.RawAccountURL = schemas.Copy(credSource.RawAccountURL).(*string)
	out.RawCredential = schemas.Copy(credSource.RawCredential).(*string)
	return out
}

// Validate implements the check.Validatable interface.
func (c AzureConfigV0) Validate() []error {
	var errs []error
	if c.RawContainer == nil {
		errs = append(errs, errors.New("'container' must not be empty"))
	}
	if c.RawConnectionString != nil && c.RawAccountURL != nil {
		errs = append(errs, errors.New(
			"exactly one of 'connection_string' or 'account_url' must be set"))
	}
	if c.RawConnectionString != nil && c.RawCredential != nil {
		errs = append(errs, errors.New(
			"'credential' and 'connection_string' must not both be set"))
	}
	return errs
}
